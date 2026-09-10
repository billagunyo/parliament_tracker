import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import pandas as pd

from database import get_cached_score, save_score_to_cache
from schemas import BillAnalysisSchema
from bill_content import get_bill_content

load_dotenv()


async def analyze_single_bill_async(
    client: AsyncOpenAI,
    bill_row: pd.Series,
    semaphore: asyncio.Semaphore,
    listing_df: pd.DataFrame,
) -> dict:
    bill_id = bill_row["bill_id"]

    cached = get_cached_score(bill_id)
    if cached:
        return cached

    # Fetch the bill's actual stated purpose (Objects and Reasons), not just
    # the tracker's procedural remarks. This runs synchronously inside the
    # semaphore-limited async worker -- fine at this concurrency level.
    content = get_bill_content(bill_id, bill_row["title"], listing_df)
    objects_and_reasons = content["objects_and_reasons"]

    if objects_and_reasons:
        substance_block = f"Stated Objects and Reasons (from the bill itself):\n{objects_and_reasons}"
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
                "content_source": content["extraction_method"],
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
                "content_source": content["extraction_method"],
            }


async def stream_pipeline(bills_df: pd.DataFrame, listing_df: pd.DataFrame, max_concurrency: int = 2):
    """
    Yields (bill_row, analysis_dict) one at a time, in the order given.
    Pass bills_df sorted most-recent-first for most-recent-first streaming.
    """
    client = AsyncOpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
    )
    semaphore = asyncio.Semaphore(max_concurrency)

    for _, row in bills_df.iterrows():
        result = await analyze_single_bill_async(client, row, semaphore, listing_df)
        yield row, result


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
