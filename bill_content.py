"""
Finds the full bill PDF (from the Bills listing page, not the tracker),
downloads it, and extracts the "MEMORANDUM OF OBJECTS AND REASONS" section
-- the part of a Kenyan bill that actually explains what it does and why.

This is what gives the LLM real substance to score, instead of just a
procedural status label like "Passed" or "Committee Stage: Pending".
"""

import re
import io
import difflib
import requests
import pandas as pd
import pdfplumber

from database import get_cached_bill_content, save_bill_content_to_cache

BASE_URL = "https://www.parliament.go.ke"
BILLS_LISTING_URL = (
    "https://www.parliament.go.ke/the-national-assembly/house-business/bills"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def fetch_bills_listing(max_pages: int = 6) -> pd.DataFrame:
    """
    Scrapes the Bills listing page (title + PDF link) across several pages.
    This is separate from the Bills Tracker -- it's where the actual bill
    documents live.
    """
    from bs4 import BeautifulSoup

    records = []
    for page_num in range(max_pages):
        url = f"{BILLS_LISTING_URL}?title=%20&field_parliament_value=2022&page={page_num}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        found_any = False

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".pdf"):
                full_url = href if href.startswith("http") else BASE_URL + href
                title_text = link.get_text(strip=True)
                if title_text:
                    records.append({"title": title_text, "pdf_url": full_url})
                    found_any = True

        if not found_any:
            break  # ran out of pages

    df = pd.DataFrame(records).drop_duplicates(subset="pdf_url")
    return df


def _normalize_title(title: str) -> str:
    """Lowercases, strips punctuation/noise words, for fuzzy matching."""
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\b(the|bill|no|amendment)\b", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def find_bill_pdf_url(tracker_title: str, listing_df: pd.DataFrame, threshold: float = 0.55) -> str | None:
    """Fuzzy-matches a tracker bill title against the bills listing to find its PDF."""
    if listing_df.empty:
        return None

    target = _normalize_title(tracker_title)
    best_score = 0.0
    best_url = None

    for _, row in listing_df.iterrows():
        candidate = _normalize_title(row["title"])
        score = difflib.SequenceMatcher(None, target, candidate).ratio()
        if score > best_score:
            best_score = score
            best_url = row["pdf_url"]

    if best_score >= threshold:
        return best_url
    return None


def extract_objects_and_reasons(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Extracts the 'MEMORANDUM OF OBJECTS AND REASONS' section from a bill PDF.
    Returns (text, method) where method is 'objects_and_reasons' if the
    section was found, or 'fallback_full_text' if we had to fall back to
    the start of the document.
    """
    full_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    combined = "\n".join(full_text)

    match = re.search(
        r"MEMORANDUM OF OBJECTS AND REASONS(.*?)(?:STATEMENT ON THE DELEGATION|PART I\b|$)",
        combined,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        section = match.group(1).strip()
        section = re.sub(r"\s+", " ", section)
        return section[:3000], "objects_and_reasons"

    # Fallback: no memorandum found, use the first chunk of the document
    fallback = re.sub(r"\s+", " ", combined)[:2000]
    return fallback, "fallback_full_text"


def get_bill_content(bill_id: str, tracker_title: str, listing_df: pd.DataFrame) -> dict:
    """
    Main entry point. Returns a dict with the bill's substantive content,
    using the SQLite cache to avoid re-downloading PDFs on repeat runs.
    """
    cached = get_cached_bill_content(bill_id)
    if cached:
        return cached

    pdf_url = find_bill_pdf_url(tracker_title, listing_df)
    if not pdf_url:
        result = {
            "pdf_url": None,
            "objects_and_reasons": "",
            "extraction_method": "no_match_found",
        }
        save_bill_content_to_cache(bill_id, "", "", "no_match_found")
        return result

    try:
        response = requests.get(pdf_url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        content, method = extract_objects_and_reasons(response.content)
    except Exception as e:
        content, method = "", f"error: {e}"

    save_bill_content_to_cache(bill_id, pdf_url, content, method)
    return {
        "pdf_url": pdf_url,
        "objects_and_reasons": content,
        "extraction_method": method,
    }
