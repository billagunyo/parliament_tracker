"""
Scrapes the Kenya National Assembly Bills Tracker PDF and parses it into
a structured DataFrame: bill title, sponsor MP, bill number, introduction
date, legislative stage, and remarks.

Source: https://www.parliament.go.ke/the-national-assembly/house-business/bill-tracker
"""

import re
import io
import requests
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from datetime import datetime

TRACKER_PAGE_URL = "https://www.parliament.go.ke/the-national-assembly/house-business/bill-tracker"
BASE_URL = "https://www.parliament.go.ke"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def find_latest_tracker_pdf_url() -> str:
    """Fetches the Bill Tracker page and returns the URL of the most recent tracker PDF."""
    response = requests.get(TRACKER_PAGE_URL, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    pdf_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf") and "tracker" in href.lower():
            full_url = href if href.startswith("http") else BASE_URL + href
            pdf_links.append(full_url)

    if not pdf_links:
        raise RuntimeError(
            "No tracker PDF links found on the Bill Tracker page. "
            "The site structure may have changed."
        )

    # The page lists trackers newest-first, so the first match is usually
    # the latest. We keep it simple and just take the first one found.
    return pdf_links[0]


def download_pdf_text(pdf_url: str) -> str:
    """Downloads a PDF and extracts all its text, page by page."""
    response = requests.get(pdf_url, headers=HEADERS, timeout=60)
    response.raise_for_status()

    full_text = []
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    return "\n".join(full_text)


def _clean_text(raw_text: str) -> str:
    """Strips repeated page headers/footers that appear on every page of the PDF."""
    # Remove repeated page banner e.g. "Status as at Thursday, 27th August 2026 The National Assembly"
    raw_text = re.sub(
        r"Status as at.*?National Assembly", "", raw_text, flags=re.DOTALL
    )
    # Remove repeated column header row
    raw_text = re.sub(
        r"S/No/\s*BILL\s*SPONSOR.*?REMARKS\s*ASSENT",
        "",
        raw_text,
        flags=re.DOTALL,
    )
    # Remove standalone page-number lines
    raw_text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", raw_text)
    return raw_text


def parse_bills_tracker_text(raw_text: str) -> pd.DataFrame:
    """
    Splits the cleaned tracker text into individual bill entries and extracts
    structured fields with regex. Best-effort — inspect the resulting CSV.
    """
    text = _clean_text(raw_text)

    # Each bill entry starts with "<number>. " at the start of a line.
    entry_pattern = re.compile(r"(?m)^(\d{1,3})\.\s")
    matches = list(entry_pattern.finditer(text))

    records = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        sno = match.group(1)

        record = _parse_single_entry(sno, chunk)
        if record:
            records.append(record)

    df = pd.DataFrame(records)
    return df


def _parse_single_entry(sno: str, chunk: str) -> dict | None:
    """Parses one bill's raw text chunk into a structured record."""

    # Bill/Senate number pattern, e.g. "NA Bill No. 34 of 2022" or "Sen. Bill No. 5 of 2023"
    bill_no_pattern = re.compile(
        r"((?:NA\.?|Sen\.?)\s*Bill\s*No\.?\s*\d+\s*of\s*\d{4})", re.IGNORECASE
    )
    bill_no_match = bill_no_pattern.search(chunk)
    if not bill_no_match:
        return None  # Skip anything we can't confidently parse

    bill_no = bill_no_match.group(1).strip()
    before_bill_no = chunk[: bill_no_match.start()].strip()
    after_bill_no = chunk[bill_no_match.end() :].strip()

    # Everything before the bill number is "Title ... Sponsor".
    # Heuristic: sponsor names/offices are typically the LAST line(s) before
    # the bill number and often contain "Hon.", "Sen.", "Leader", "Chairperson",
    # "Deputy Speaker", or end in ", MP".
    lines = [l.strip() for l in before_bill_no.split("\n") if l.strip()]

    sponsor_keywords = (
        "hon.", "sen.", "leader of", "chairperson", "deputy speaker",
        "speaker", "the senate majority", "the senate minority",
    )

    title_lines = []
    sponsor_lines = []
    seen_sponsor_start = False
    for line in lines:
        lower = line.lower()
        looks_like_sponsor = any(k in lower for k in sponsor_keywords) or line.endswith(", MP")
        if looks_like_sponsor:
            seen_sponsor_start = True
        if seen_sponsor_start:
            sponsor_lines.append(line)
        else:
            title_lines.append(line)

    title = " ".join(title_lines).strip(" ,.")
    sponsor = " ".join(sponsor_lines).strip(" ,.") or "Unknown"

    # First date after the bill number = "DATED" (introduction date)
    date_pattern = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
    date_match = date_pattern.search(after_bill_no)
    dated_str = date_match.group(1) if date_match else None
    dated = _parse_date(dated_str)

    # Stage / status heuristics based on keywords present in the remaining text
    lower_after = after_bill_no.lower()
    if "assent" in lower_after and re.search(r"\d{1,2}/\d{1,2}/\d{4}", after_bill_no.split("assent")[-1] if "assent" in lower_after else ""):
        stage = "Assented"
    elif "lapsed" in lower_after:
        stage = "Lapsed"
    elif "withdrawn" in lower_after:
        stage = "Withdrawn"
    elif "lost" in lower_after:
        stage = "Lost"
    elif "passed" in lower_after:
        stage = "Passed"
    elif "committee stage" in lower_after:
        stage = "Committee"
    elif "2nd read" in lower_after or "2ⁿᵈ read" in lower_after:
        stage = "2nd Reading"
    elif "1st read" in lower_after or "1ˢᵗ read" in lower_after:
        stage = "1st Reading"
    else:
        stage = "Unknown"

    remarks = after_bill_no.replace("\n", " ").strip()
    remarks = re.sub(r"\s+", " ", remarks)[:500]  # cap length

    return {
        "sno": sno,
        "bill_id": bill_no.replace(" ", "_").replace(".", ""),
        "title": title,
        "sponsor": sponsor,
        "bill_no": bill_no,
        "dated": dated,
        "stage": stage,
        "remarks": remarks,
    }


def _parse_date(date_str: str | None):
    if not date_str:
        return None
    for fmt in ("%d/%m/%Y",):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def fetch_parliament_bills() -> pd.DataFrame:
    """
    Main entry point: finds the latest Bills Tracker PDF, downloads it,
    parses it, and returns a DataFrame sorted most-recent-first.
    """
    pdf_url = find_latest_tracker_pdf_url()
    raw_text = download_pdf_text(pdf_url)
    df = parse_bills_tracker_text(raw_text)

    # Save raw parsed data for manual QA
    df.to_csv("bills_scraped_raw.csv", index=False)

    # Drop rows with no parseable date, sort most-recent-first
    df = df.dropna(subset=["dated"])
    df = df.sort_values(by="dated", ascending=False).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Quick manual test: run `python scraper.py` to sanity-check the scraper
    # before wiring it into the Streamlit app.
    bills_df = fetch_parliament_bills()
    print(f"Parsed {len(bills_df)} bills.")
    print(bills_df.head(10).to_string())
