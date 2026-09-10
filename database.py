import sqlite3

DB_NAME = "bill_cache.db"


def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bill_scores (
                bill_id TEXT PRIMARY KEY,
                impact_score INTEGER,
                economic_impact INTEGER,
                social_impact INTEGER,
                llm_summary TEXT,
                justification TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def get_cached_score(bill_id: str) -> dict | None:
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT impact_score, economic_impact, social_impact, llm_summary, justification 
            FROM bill_scores 
            WHERE bill_id = ?
            """,
            (bill_id,),
        )
        row = cursor.fetchone()

        if row:
            return {
                "bill_id": bill_id,
                "Impact_Score": row[0],
                "Economic_Impact": row[1],
                "Social_Impact": row[2],
                "LLM_Summary": row[3],
                "Justification": row[4],
                "status": "cached",
            }
        return None


def save_score_to_cache(bill_id: str, analysis_data: dict):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO bill_scores (bill_id, impact_score, economic_impact, social_impact, llm_summary, justification)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                bill_id,
                analysis_data["Impact_Score"],
                analysis_data["Economic_Impact"],
                analysis_data["Social_Impact"],
                analysis_data["LLM_Summary"],
                analysis_data["Justification"],
            ),
        )
        conn.commit()