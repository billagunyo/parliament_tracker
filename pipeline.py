import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import pandas as pd

from database import get_cached_score, save_score_to_cache
from schemas import BillAnalysisSchema
from bill_content import get_bill_content

load_dotenv()


# pipeline.py — add this alongside the existing async code
import openai  # sync client

def analyze_single_bill_sync(bill_row, listing_df):
    """Synchronous single-bill analyzer for Streamlit. No asyncio, no deadlocks."""
    bill_id = bill_row["bill_id"]

    cached = get_cached_score(bill_id)
    if cached:
        return cached

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

    client = openai.OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
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
