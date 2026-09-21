"""
scripts/train_transformer_model.py
----------------------------------
Trains a contextual embedding-based Multi-Label classifier for Mindify using
SentenceTransformer ('all-MiniLM-L6-v2') + MultiOutputClassifier(LogisticRegression).
Saves models/transformer_classifier.pkl and models/label_binarizer.pkl.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "multilabel_training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_classifier.pkl")
BINARIZER_PATH = os.path.join(MODEL_DIR, "label_binarizer.pkl")


def train_transformer_model():
    print(f"[Transformer Trainer] Loading dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please generate dataset first.")

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels_list = [str(labels).split("|") for labels in df["labels"]]

    # Encode labels using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(labels_list)
    print(f"[Transformer Trainer] Binarized {len(df)} samples across {len(mlb.classes_)} unique label classes.")

    # Load SentenceTransformer model
    print("[Transformer Trainer] Loading SentenceTransformer('all-MiniLM-L6-v2')...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute dense contextual embeddings
    print(f"[Transformer Trainer] Encoding {len(texts)} texts into contextual embeddings...")
    X_embeddings = embedder.encode(texts, show_progress_bar=True)

    # Build & train MultiOutputClassifier(CalibratedClassifierCV(LogisticRegression(), cv=3))
    print("[Transformer Trainer] Training MultiOutputClassifier(CalibratedClassifierCV(LogisticRegression(), cv=3))...")
    base_estimator = CalibratedClassifierCV(
        estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        cv=3
    )
    clf = MultiOutputClassifier(base_estimator)
    clf.fit(X_embeddings, y_encoded)

    # Save ONLY clf and MultiLabelBinarizer via joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(mlb, BINARIZER_PATH)
    print(f"[Transformer Trainer] Saved transformer classifier model to: {MODEL_PATH}")
    print(f"[Transformer Trainer] Saved label binarizer instance to: {BINARIZER_PATH}")


if __name__ == "__main__":
    train_transformer_model()
