"""
scripts/train_model.py
----------------------
Trains and optimizes a Scikit-Learn TF-IDF + LinearSVC intent classification pipeline
using 5-Fold Stratified Cross-Validation & GridSearchCV across the 36-category taxonomy.
Evaluates the best estimator on a holdout test split and exports to models/intent_classifier.pkl.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier.pkl")


def train_intent_model():
    print(f"[Model Trainer] Loading dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run scripts/generate_dataset.py first.")

    df = pd.read_csv(DATA_PATH)
    X = df["text"].astype(str)
    y = df["category"].astype(str)

    print(f"[Model Trainer] Dataset contains {len(df)} samples across {df['category'].nunique()} categories.")

    # 80/20 train-test split with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Build Scikit-Learn Pipeline combining TfidfVectorizer and LinearSVC
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=1)),
        ('linearsvc', LinearSVC(class_weight='balanced', random_state=42))
    ])

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
        'tfidf__sublinear_tf': [True, False],
        'linearsvc__C': [0.1, 1.0, 10.0]
    }

    print("[Model Trainer] Initializing 5-Fold GridSearchCV across parameter grid...")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    print("[Model Trainer] Fitting GridSearchCV on training data...")
    grid.fit(X_train, y_train)

    print("\n" + "=" * 65)
    print("                 GRIDSEARCHCV OPTIMIZATION RESULTS")
    print("=" * 65)
    print(f"Best Hyperparameters : {grid.best_params_}")
    print(f"Best Cross-Val Score  : {grid.best_score_:.4f}")
    print("=" * 65 + "\n")

    # Evaluate best estimator on holdout test set
    y_pred = grid.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\n" + "=" * 65)
    print("      CLASSIFICATION REPORT (HOLDOUT TEST EVALUATION)")
    print("=" * 65)
    print(report)
    print("=" * 65 + "\n")

    # Save best estimator to disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"[Model Trainer] Saved best model estimator to: {MODEL_PATH}")


if __name__ == "__main__":
    train_intent_model()
