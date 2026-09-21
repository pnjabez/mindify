"""
scripts/train_complex_model.py
------------------------------
Trains a Scikit-Learn Multi-Label NLP intent classification pipeline using
TfidfVectorizer + OneVsRestClassifier(LogisticRegression) & MultiLabelBinarizer.
Evaluates accuracy via hamming_loss and classification_report, then exports to
models/multilabel_classifier.pkl and models/label_binarizer.pkl.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "multilabel_training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "multilabel_classifier.pkl")
BINARIZER_PATH = os.path.join(MODEL_DIR, "label_binarizer.pkl")


def train_multilabel_model():
    print(f"[Multi-Label Trainer] Loading dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run scripts/generate_complex_dataset.py first.")

    df = pd.read_csv(DATA_PATH)
    X = df["text"].astype(str)

    # Parse comma-separated label strings into list of list of label strings
    labels_list = [str(labels).split(",") for labels in df["labels"]]

    # Binarize target labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_list)

    print(f"[Multi-Label Trainer] Dataset contains {len(df)} samples across {len(mlb.classes_)} unique label classes.")

    # 80/20 train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )

    # Build Pipeline with TfidfVectorizer & OneVsRestClassifier(LogisticRegression)
    print("[Multi-Label Trainer] Initializing TF-IDF + OneVsRestClassifier(LogisticRegression) pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ('clf', OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)))
    ])

    # Fit pipeline
    print("[Multi-Label Trainer] Training multi-label classifier...")
    pipeline.fit(X_train, Y_train)

    # Evaluate on holdout test set
    Y_pred = pipeline.predict(X_test)
    h_loss = hamming_loss(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, target_names=mlb.classes_, zero_division=0)

    print("\n" + "=" * 65)
    print("      MULTI-LABEL CLASSIFICATION EVALUATION METRICS")
    print("=" * 65)
    print(f"Hamming Loss (lower is better) : {h_loss:.4f}")
    print(f"Exact Subset Accuracy           : {(Y_pred == Y_test).all(axis=1).mean():.4f}")
    print("=" * 65)
    print(report)
    print("=" * 65 + "\n")

    # Fit pipeline on 100% of data before saving for production inference
    pipeline.fit(X, Y)

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(mlb, BINARIZER_PATH)
    print(f"[Multi-Label Trainer] Saved trained classifier model to: {MODEL_PATH}")
    print(f"[Multi-Label Trainer] Saved label binarizer instance to: {BINARIZER_PATH}")


if __name__ == "__main__":
    train_multilabel_model()
