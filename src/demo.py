"""Interactive demo: print top-1 and top-3 predicted support categories."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from src.preprocess import clean_text

MODEL_PATH = Path("results/logistic_regression_tfidf.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict support category for a question.")
    parser.add_argument("question", nargs="+", help="Student question text")
    args = parser.parse_args()
    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit("Please provide a non-empty question.")

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing {MODEL_PATH}. Train first: python -m src.train_baseline"
        )

    model = joblib.load(MODEL_PATH)
    cleaned = clean_text(question)
    proba = model.predict_proba([cleaned])[0]
    classes = np.asarray(model.classes_)
    order = np.argsort(-proba)
    top1 = classes[order[0]]
    top3 = [(str(classes[i]), float(proba[i])) for i in order[:3]]

    print("Question:", question)
    print("Top-1 category:", top1)
    print("Top-3 (category, confidence):")
    for cat, score in top3:
        print(f"  {cat}: {score:.4f}")


if __name__ == "__main__":
    main()
