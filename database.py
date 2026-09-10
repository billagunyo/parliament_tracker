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
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bill_content (
                bill_id TEXT PRIMARY KEY,
                pdf_url TEXT,
                objects_and_reasons TEXT,
                extraction_method TEXT,
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


def get_cached_bill_content(bill_id: str) -> dict | None:
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT pdf_url, objects_and_reasons, extraction_method
            FROM bill_content
            WHERE bill_id = ?
            """,
            (bill_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "pdf_url": row[0],
                "objects_and_reasons": row[1],
                "extraction_method": row[2],
            }
        return None


def save_bill_content_to_cache(bill_id: str, pdf_url: str, content: str, method: str):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO bill_content (bill_id, pdf_url, objects_and_reasons, extraction_method)
            VALUES (?, ?, ?, ?)
            """,
            (bill_id, pdf_url, content, method),
        )
        conn.commit()
