import streamlit as st
import pandas as pd

from database import init_db
from pipeline import compute_mp_rankings, analyze_single_bill_sync
from scraper import fetch_parliament_bills
from bill_content import fetch_bills_listing

st.set_page_config(page_title="Parliamentary MP Impact Tracker", layout="wide")

init_db()

st.title("Parliamentary MP & Bill Impact Tracker")

st.markdown(
    """
This application scrapes the **live Kenya National Assembly Bills Tracker** for
status and sponsor data, cross-references each bill against its **actual PDF**
to pull the stated *Objects and Reasons*, and scores real-world impact on that
substance -- not just the procedural status label.
"""
)

max_bills = st.slider(
    "Number of most-recent bills to process", min_value=5, max_value=30, value=10
)

if st.button("Fetch Bills & Run Scoring Pipeline"):

    # ---- Step 1: tracker scrape ----
    with st.spinner("Scraping the Bills Tracker (status/sponsor data)..."):
        try:
            all_bills = fetch_parliament_bills()
        except Exception as e:
            st.error(f"Failed to scrape the tracker: {e}")
            st.stop()

    if all_bills.empty:
        st.error("No bills could be parsed from the tracker PDF.")
        st.stop()

    # ---- Step 2: bills listing ----
    with st.spinner("Scraping the Bills listing (for actual bill PDFs)..."):
        try:
            listing_df = fetch_bills_listing(max_pages=6)
        except Exception as e:
            st.warning(f"Could not fetch the bills listing ({e}). Scores will fall back to lower confidence.")
            listing_df = pd.DataFrame(columns=["title", "pdf_url"])

    st.success(
        f"Scraped {len(all_bills)} tracked bills and {len(listing_df)} listed bill documents. "
        f"Processing the {max_bills} most recent..."
    )

    bills_to_process = all_bills.head(max_bills)

    # ---- Step 3: per-bill loop with LIVE incremental UI ----
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    results_area = st.container()

    collected_rows = []
    total = len(bills_to_process)

    for i, (_, bill_row) in enumerate(bills_to_process.iterrows(), start=1):
        status_text.info(f"⏳ [{i}/{total}] Analyzing: {bill_row['title'][:80]}...")

        try:
            analysis = analyze_single_bill_sync(bill_row, listing_df)
        except Exception as e:
            analysis = {
                "bill_id": bill_row["bill_id"],
                "Impact_Score": 0,
                "Economic_Impact": 0,
                "Social_Impact": 0,
                "LLM_Summary": "Failed.",
                "Justification": str(e),
                "status": "failed",
                "content_source": "error",
            }

        merged = {**bill_row.to_dict(), **analysis}
        collected_rows.append(merged)

        score = analysis["Impact_Score"]
        emoji = "🟢" if score > 0 else ("🔴" if score < 0 else "⚪")
        source_note = (
            "📄 real bill text"
            if analysis.get("content_source") == "objects_and_reasons"
            else "⚠️ low confidence, no bill text matched"
        )
        with results_area:
            st.write(
                f"{emoji} **{bill_row['title']}** "
                f"— Sponsor: {bill_row['sponsor']} "
                f"— Stage: {bill_row['stage']} "
                f"— Score: {score} "
                f"— {source_note}"
            )

        progress_bar.progress(i / total)

    status_text.success(f"✅ Done. Analyzed {len(collected_rows)} bills.")

    # ---- Step 4: aggregation ----
    processed_bills = pd.DataFrame(collected_rows)
    mp_leaderboard = compute_mp_rankings(processed_bills)

    st.divider()

    matched = len(processed_bills[processed_bills.get("content_source") == "objects_and_reasons"])
    st.info(f"{matched} of {len(processed_bills)} bills were scored using their actual bill text.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Bills Processed", len(processed_bills))
    if not mp_leaderboard.empty:
        c2.metric("Top Ranked MP/Sponsor", mp_leaderboard.iloc[0]["sponsor"])
        c3.metric("Top Score", f"{mp_leaderboard.iloc[0]['Net_Positive_Score']} pts")

    st.subheader("MP / Sponsor Legislative Impact Leaderboard")
    st.dataframe(mp_leaderboard, use_container_width=True)

    st.subheader("Net Impact Score by Sponsor")
    if not mp_leaderboard.empty:
        st.bar_chart(data=mp_leaderboard, x="sponsor", y="Net_Positive_Score")

    st.subheader("📑 Detailed Bill Analysis")
    st.dataframe(
        processed_bills[
            [
                "title", "sponsor", "stage", "dated", "status", "content_source",
                "Impact_Score", "LLM_Summary", "Justification",
            ]
        ],
        use_container_width=True,
    )

    st.divider()
    st.subheader("Download Data")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download MP Leaderboard (CSV)",
            data=mp_leaderboard.to_csv(index=False).encode("utf-8"),
            file_name="mp_leaderboard.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label="Download Bill-Level Data (CSV)",
            data=processed_bills.to_csv(index=False).encode("utf-8"),
            file_name="bill_analysis.csv",
            mime="text/csv",
        )
