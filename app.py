import asyncio
import pandas as pd
import streamlit as st

from database import init_db
from pipeline import run_pipeline

st.set_page_config(page_title="Parliamentary MP Impact Tracker", layout="wide")

init_db()

st.title("Parliamentary MP & Bill Impact Tracker")

st.markdown(
    """
This application processes parliamentary bills concurrently using free OpenAI models via **GitHub Models**,
caches outputs to SQLite, and computes a net impact leaderboard for MPs.
"""
)

mock_bills = pd.DataFrame(
    [
        {
            "bill_id": "B001",
            "title": "Digital Tax Reform & Infrastructure Act",
            "text_summary": "Introduces a 1.5% digital service tax while funding broadband expansion in rural constituencies.",
            "sponsor": "John Doe",
            "party": "Party A",
            "stage": "2nd Reading",
        },
        {
            "bill_id": "B002",
            "title": "Universal Healthcare Subsidies Bill",
            "text_summary": "Provides direct financial subsidies for primary healthcare and essential chronic medication.",
            "sponsor": "Jane Smith",
            "party": "Party B",
            "stage": "Assented",
        },
        {
            "bill_id": "B003",
            "title": "Public Procurement Transparency Amendment",
            "text_summary": "Mandates open public disclosure of all government procurement awards above 5 Million.",
            "sponsor": "John Doe",
            "party": "Party A",
            "stage": "1st Reading",
        },
        {
            "bill_id": "B004",
            "title": "Agricultural Export Processing Licensing Tax",
            "text_summary": "Increases compliance levies and documentation costs for smallholder agricultural exporters.",
            "sponsor": "Alice Johnson",
            "party": "Party C",
            "stage": "Committee",
        },
    ]
)

if st.button("Run AI Scoring & MP Ranking Pipeline"):
    with st.spinner("Processing bills via GitHub Models & SQLite Cache..."):
        processed_bills, mp_leaderboard = asyncio.run(
            run_pipeline(mock_bills, max_concurrency=2)
        )

    cache_hits = len(processed_bills[processed_bills["status"] == "cached"])
    live_calls = len(processed_bills[processed_bills["status"] == "live_api"])

    st.success(
        f"Pipeline Run Complete! ({cache_hits} Loaded from Cache | {live_calls} Live API Calls)"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Bills Processed", len(processed_bills))
    c2.metric("Top Ranked MP", mp_leaderboard.iloc[0]["sponsor"])
    c3.metric(
        "Top Score", f"{mp_leaderboard.iloc[0]['Net_Positive_Score']} pts"
    )

    st.divider()

    st.subheader("MP Legislative Impact Leaderboard")
    st.dataframe(mp_leaderboard, use_container_width=True)

    st.subheader("Net Impact Score by MP")
    st.bar_chart(data=mp_leaderboard, x="sponsor", y="Net_Positive_Score")

    st.divider()

    st.subheader("Detailed Bill Analysis & Cache Status")
    st.dataframe(
        processed_bills[
            [
                "bill_id",
                "title",
                "sponsor",
                "stage",
                "status",
                "Impact_Score",
                "LLM_Summary",
                "Justification",
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