"""
backend/db.py
-------------
SQLite Database helper module for Mindify. Handles persistence of generated
affirmations, implementation intentions, micro-actions, sentiment scores,
and linguistic distortions.
"""

import json
import os
import sqlite3

# ---------------------------------------------------------------------------
# Database File Location
# Path resolves to <project_root>/data/mindify.db
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "mindify.db")


def get_connection() -> sqlite3.Connection:
    """Connect to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the SQLite database and create the `history` table if it
    does not already exist. Creates the `data/` directory if missing.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                goal TEXT,
                affirmation TEXT,
                intention TEXT,
                micro_action TEXT,
                sentiment REAL,
                distortions TEXT
            )
            """
        )
        conn.commit()
    print(f"[Mindify DB] Database initialized at: {DB_PATH}", flush=True)


def save_entry(data_dict: dict) -> int:
    """
    Insert a processed NLP result dictionary into the `history` table.

    Args:
        data_dict (dict): Result dictionary containing NLP keys.

    Returns:
        int: The inserted record's row ID.
    """
    goal = data_dict.get("original_goal", "")
    affirmation = data_dict.get("generated_affirmation", "")
    intention = data_dict.get("implementation_intention", "")
    micro_action = data_dict.get("micro_action", "")
    sentiment = data_dict.get("sentiment", 0.0)

    # Convert distortions list or dict to JSON string for storage
    raw_distortions = data_dict.get("distortions", [])
    distortions_json = json.dumps(raw_distortions)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO history (goal, affirmation, intention, micro_action, sentiment, distortions)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (goal, affirmation, intention, micro_action, sentiment, distortions_json),
        )
        conn.commit()
        return cursor.lastrowid


def get_history() -> list[dict]:
    """
    Fetch all stored manifestation records ordered by timestamp descending.

    Returns:
        list[dict]: List of history records with distortions deserialized back to Python list.
    """
    if not os.path.exists(DB_PATH):
        init_db()

    results = []
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, goal, affirmation, intention, micro_action, sentiment, distortions
            FROM history
            ORDER BY id DESC
            """
        )
        rows = cursor.fetchall()

        for row in rows:
            raw_dist = row["distortions"]
            parsed_dist = []
            if raw_dist:
                try:
                    parsed_dist = json.loads(raw_dist)
                except (json.JSONDecodeError, TypeError):
                    parsed_dist = []

            results.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "goal": row["goal"],
                    "affirmation": row["affirmation"],
                    "intention": row["intention"],
                    "micro_action": row["micro_action"],
                    "sentiment": row["sentiment"],
                    "distortions": parsed_dist,
                }
            )

    return results
