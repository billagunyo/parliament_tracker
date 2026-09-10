import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import pandas as pd

from database import get_cached_score, save_score_to_cache
from schemas import BillAnalysisSchema

load_dotenv()


async def analyze_single_bill_async(
    client: AsyncOpenAI,
    bill_row: pd.Series,
    semaphore: asyncio.Semaphore,
) -> dict:
    bill_id = bill_row["bill_id"]

    # 1. Check local cache
    cached = get_cached_score(bill_id)
    if cached:
        return cached

    # 2. Call GitHub Models API
    system_prompt = (
        "You are an objective legislative policy analyst. Evaluate the proposed bill text "
        "and score its overall public impact based on economic burden, rights protection, "
        "and public service delivery. Return valid JSON matching the schema."
    )
    user_prompt = f"Bill Title: {bill_row['title']}\n\nSummary Text:\n{bill_row.get('text_summary', bill_row['title'])}"

    async with semaphore:
        try:
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=BillAnalysisSchema,
            )

            result: BillAnalysisSchema = response.choices[0].message.parsed

            analysis_data = {
                "bill_id": bill_id,
                "Impact_Score": result.overall_impact_score,
                "Economic_Impact": result.economic_impact,
                "Social_Impact": result.social_impact,
                "LLM_Summary": result.summary,
                "Justification": result.impact_justification,
                "status": "live_api",
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
            }


async def run_pipeline(
    bills_df: pd.DataFrame, max_concurrency: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = AsyncOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        analyze_single_bill_async(client, row, semaphore)
        for _, row in bills_df.iterrows()
    ]
    results = await asyncio.gather(*tasks)

    analysis_df = pd.DataFrame(results)
    merged_df = pd.merge(bills_df, analysis_df, on="bill_id")

    stage_weights = {
        "1st Reading": 1.0,
        "2nd Reading": 1.2,
        "Committee": 1.5,
        "Assented": 2.0,
    }
    merged_df["Stage_Weight"] = (
        merged_df["stage"].map(stage_weights).fillna(1.0)
    )
    merged_df["Weighted_Score"] = (
        merged_df["Impact_Score"] * merged_df["Stage_Weight"]
    )

    mp_rankings = (
        merged_df.groupby(["sponsor", "party"])
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
    )

    return merged_df, mp_rankings