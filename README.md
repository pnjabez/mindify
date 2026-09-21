# 🧠 Mindify – Smart Affirmation & Goal Generator

**Mindify** is a 100% local, privacy-first Natural Language Processing (NLP) web application that converts raw goals and manifestations into structured, science-backed affirmations, implementation intentions, micro-actions, and emotional resilience statements.

---

## ✨ Features

- **5-Phase Local NLP Pipeline**:
  1. **Lexical Processing**: Automated spellchecking and tokenization via `pyspellchecker`.
  2. **Syntactic Processing**: Subject-Verb-Object (S-V-O) dependency parsing via `spaCy` to determine Locus of Control (Internal vs. External).
  3. **Semantic Processing**: Multi-label intent classification via `scikit-learn` & `sentence-transformers` across 90+ life taxonomy categories.
  4. **Discourse Processing**: TextRank sentence summarization via `pytextrank` and sentiment analysis using `NLTK VADER`.
  5. **Pragmatic Processing**: Temporal focus detection (rumination vs. anticipatory anxiety) and NLG modular assembly for generating "Even if..." resilience statements.
- **Privacy First**: Runs 100% locally on your machine with zero external API dependencies or cloud telemetry.
- **Interactive UI & Inspector**: Includes a clean, modern UI with an inline Developer Diagnostics Inspector.
- **Persistence & History**: Local SQLite database storing history, sentiment scores, and detected cognitive patterns.

---

## 📁 Project Structure

```text
mindify_project/
├── backend/
│   ├── app.py            # Flask API entry point
│   └── db.py             # SQLite database helper module
├── NLP/
│   └── processor.py      # Core 5-phase NLP pipeline logic
├── frontend/
│   ├── static/           # CSS stylesheets and JavaScript
│   └── templates/        # HTML templates (index.html)
├── models/               # Trained Scikit-Learn & Transformer ML models (.pkl)
├── data/                 # Datasets (.csv) and SQLite database (mindify.db)
├── scripts/              # Dataset generation and model training scripts
├── requirements.txt      # Python dependencies
└── .gitignore            # Git exclusion rules
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed on your machine.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/mindify.git
   cd mindify
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python backend/app.py
   ```

5. **Open in browser**:
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## 🛠️ Training the NLP Models

If you want to regenerate datasets or retrain the intent classification models:

```bash
# Generate synthetic training datasets
python scripts/generate_dataset.py

# Train TF-IDF + LinearSVC single-intent classifier
python scripts/train_model.py

# Train SentenceTransformer + MultiOutputClassifier multi-label classifier
python scripts/train_transformer_model.py
```

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for details.
