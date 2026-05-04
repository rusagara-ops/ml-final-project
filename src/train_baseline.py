"""Train TF-IDF + Logistic Regression baseline and save model + metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluate import evaluate_model

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODEL_PATH = RESULTS_DIR / "logistic_regression_tfidf.joblib"
METRICS_PATH = RESULTS_DIR / "metrics_baseline.json"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run: python -m src.split_data --input data/sample_questions.csv"
        )

    train_df = pd.read_csv(train_path)
    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(str)

    model = build_pipeline()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    metrics: dict = {"model_path": str(MODEL_PATH)}
    metrics["train"] = evaluate_model(model, X_train, y_train)

    if val_path.exists():
        val_df = pd.read_csv(val_path)
        metrics["validation"] = evaluate_model(
            model, val_df["text"].astype(str), val_df["label"].astype(str)
        )
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        metrics["test"] = evaluate_model(
            model, test_df["text"].astype(str), test_df["label"].astype(str)
        )

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
