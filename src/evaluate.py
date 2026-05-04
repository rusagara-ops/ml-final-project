"""Evaluation helpers for multiclass text classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


def compute_top3_accuracy(model: Any, X, y) -> float | None:
    """Fraction of samples whose true label is in the top-3 by probability.

    Returns None if ``predict_proba`` is not available.
    """
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    top3_idx = np.argsort(-proba, axis=1)[:, :3]
    top3_labels = classes[top3_idx]
    y_arr = np.asarray(y)
    return float(np.mean(np.any(top3_labels == y_arr[:, None], axis=1)))


def evaluate_model(model: Any, X, y) -> dict[str, Any]:
    """Return accuracy, macro/weighted F1, per-class metrics, and top-3 accuracy."""
    y_pred = model.predict(X)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": classification_report(
            y, y_pred, output_dict=True, zero_division=0
        ),
        "top3_accuracy": compute_top3_accuracy(model, X, y),
    }
    return out
