"""
backend/app.py
--------------
Flask application entry point for Mindify.

Serves the frontend from the custom `frontend/` directory tree and
exposes REST endpoints that delegate to the NLP processor and local SQLite DB.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup – make the project root importable so that `NLP.processor`
# and `backend.db` can be found regardless of where the server is launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify
from NLP.processor import process_text
from backend.db import init_db, save_entry, get_history

# ---------------------------------------------------------------------------
# App configuration & Database initialization
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "frontend", "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "frontend", "static"),
)

# Initialize local SQLite database on startup
init_db()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the main Mindify UI."""
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Accepts a JSON payload with a 'manifestation' key, passes the text
    through the NLP pipeline, saves the result to SQLite, and returns the structured result.

    Request body (JSON):
        { "manifestation": "<user goal text>" }

    Response body (JSON):
        {
            "original_goal": "...",
            "generated_affirmation": "...",
            "implementation_intention": "...",
            "micro_action": "..."
        }
    """
    data = request.get_json(silent=True)

    # --- Input validation ---
    if not data or "manifestation" not in data:
        return jsonify({"error": "Missing 'manifestation' field in request body."}), 400

    user_text = data["manifestation"].strip()
    if not user_text:
        return jsonify({"error": "The 'manifestation' field cannot be empty."}), 400

    # --- Delegate to NLP pipeline ---
    result = process_text(user_text)

    # --- Save result to SQLite DB ---
    try:
        save_entry(result)
    except Exception as db_exc:
        print(f"[Mindify DB Warning] Failed to save entry: {db_exc}", flush=True)

    return jsonify(result), 200


@app.route("/api/history", methods=["GET"])
def history():
    """
    Fetch all historical manifestation entries from local SQLite DB.

    Response body (JSON):
        [
            {
                "id": 1,
                "timestamp": "2026-08-28 09:50:00",
                "goal": "...",
                "affirmation": "...",
                "intention": "...",
                "micro_action": "...",
                "sentiment": 0.0,
                "distortions": []
            },
            ...
        ]
    """
    entries = get_history()
    return jsonify(entries), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
