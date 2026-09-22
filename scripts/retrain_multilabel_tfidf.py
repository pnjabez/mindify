"""
scripts/retrain_multilabel_tfidf.py
------------------------------------
Retrains the multilabel classifier WITHOUT SentenceTransformer.
Uses TF-IDF + MultiOutputClassifier(LogisticRegression) instead.
This saves ~400MB of RAM at runtime — making the app fit on free cloud tiers.
Output: models/transformer_classifier.pkl and models/label_binarizer.pkl
         (same filenames so no other code needs changing).
"""

import os
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "multilabel_training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_classifier.pkl")
BINARIZER_PATH = os.path.join(MODEL_DIR, "label_binarizer.pkl")


def retrain():
    print(f"[TF-IDF Multilabel Trainer] Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels_list = [str(labels).split("|") for labels in df["labels"]]

    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(labels_list)
    print(f"[TF-IDF Multilabel Trainer] {len(df)} samples, {len(mlb.classes_)} classes.")

    base_estimator = CalibratedClassifierCV(
        estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        cv=3
    )

    # TF-IDF pipeline — no PyTorch, no GPU, tiny RAM footprint
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=15000)),
        ('clf', MultiOutputClassifier(base_estimator))
    ])

    print("[TF-IDF Multilabel Trainer] Training pipeline...")
    pipeline.fit(texts, y_encoded)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(mlb, BINARIZER_PATH)
    print(f"[TF-IDF Multilabel Trainer] Saved to: {MODEL_PATH}")
    print(f"[TF-IDF Multilabel Trainer] Saved binarizer to: {BINARIZER_PATH}")


if __name__ == "__main__":
    retrain()
