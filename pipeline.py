import os
import openai
import pandas as pd
from dotenv import load_dotenv

from database import get_cached_score, save_score_to_cache
from schemas import BillAnalysisSchema
from bill_content import get_bill_content

load_dotenv()


def analyze_single_bill_sync(bill_row: pd.Series, listing_df: pd.DataFrame) -> dict:
    """Synchronous single-bill analyzer for Streamlit. No asyncio, no deadlocks."""
    bill_id = bill_row["bill_id"]

    cached = get_cached_score(bill_id)
    if cached:
        return cached

    try:
        content = get_bill_content(bill_id, bill_row["title"], listing_df)
    except Exception as e:
        content = {
            "pdf_url": None,
            "objects_and_reasons": "",
            "extraction_method": f"content_error: {e}",
        }

    objects_and_reasons = content.get("objects_and_reasons", "")

    if objects_and_reasons:
        substance_block = (
            f"Stated Objects and Reasons (from the bill itself):\n{objects_and_reasons}"
        )
    else:
        substance_block = (
            "No bill text could be matched or extracted for this entry -- "
            "score conservatively and flag low confidence in your justification."
        )

    system_prompt = (
        "You are an independent legislative policy analyst working to help ordinary "
        "citizens evaluate bills on their real-world merits, as a counterweight to "
        "well-resourced lobbying narratives. Base your assessment on the bill's stated "
        "objects and reasons -- what it actually proposes to change -- not on its title "
        "or procedural status. If no substantive text is available, say so explicitly "
        "in your justification and score conservatively (closer to 0) rather than guessing."
    )
    user_prompt = (
        f"Bill Title: {bill_row['title']}\n"
        f"Sponsor: {bill_row['sponsor']}\n"
        f"Legislative Stage: {bill_row['stage']}\n\n"
        f"{substance_block}"
    )

    api_key = os.getenv("GITHUB_TOKEN")
    if not api_key:
        return {
            "bill_id": bill_id,
            "Impact_Score": 0,
            "Economic_Impact": 0,
            "Social_Impact": 0,
            "LLM_Summary": "Missing GITHUB_TOKEN.",
            "Justification": "GITHUB_TOKEN env var is not set.",
            "status": "failed",
            "content_source": content.get("extraction_method", "unknown"),
        }

    client = openai.OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=api_key,
    )

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=BillAnalysisSchema,
        )
        result = response.choices[0].message.parsed

        analysis_data = {
            "bill_id": bill_id,
            "Impact_Score": result.overall_impact_score,
            "Economic_Impact": result.economic_impact,
            "Social_Impact": result.social_impact,
            "LLM_Summary": result.summary,
            "Justification": result.impact_justification,
            "status": "live_api",
            "content_source": content.get("extraction_method", "unknown"),
        }
        save_score_to_cache(bill_id, analysis_data)
        return analysis_data

    except Exception as e:
        return {
            "bill_id": bill_id,
            "Impact_Score": 0,
            "Economic_Impact": 0,
            "Social_Impact": 0,
            "LLM_Summary": "Error processing bill.",
            "Justification": str(e),
            "status": "failed",
            "content_source": content.get("extraction_method", "unknown"),
        }


def build_stage_weights() -> dict:
    return {
        "1st Reading": 1.0,
        "2nd Reading": 1.2,
        "Committee": 1.5,
        "Passed": 1.8,
        "Assented": 2.0,
        "Lapsed": 0.5,
        "Withdrawn": 0.3,
        "Lost": 0.3,
        "Unknown": 1.0,
    }


def compute_mp_rankings(processed_bills: pd.DataFrame) -> pd.DataFrame:
    df = processed_bills.copy()

    stage_weights = build_stage_weights()
    df["Stage_Weight"] = df["stage"].map(stage_weights).fillna(1.0)
    df["Weighted_Score"] = df["Impact_Score"] * df["Stage_Weight"]

    mp_rankings = (
        df.groupby("sponsor")
        .agg(
            Total_Bills=("bill_id", "count"),
            Net_Positive_Score=("Weighted_Score", "sum"),
        )
        .reset_index()
    )
    mp_rankings["Score_Per_Bill"] = (
        mp_rankings["Net_Positive_Score"] / mp_rankings["Total_Bills"]
    ).round(2)
    mp_rankings = mp_rankings.sort_values(
        by="Net_Positive_Score", ascending=False
    ).reset_index(drop=True)

    return mp_rankings
