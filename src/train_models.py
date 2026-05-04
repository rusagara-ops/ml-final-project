"""Placeholder for extended model comparisons (Naive Bayes, Linear SVM).

Abel: extend this module to train and compare additional models against the
TF-IDF + Logistic Regression baseline. Suggested direction:

- ``sklearn.naive_bayes.MultinomialNB`` — works well with TF-IDF counts;
  you may need ``TfidfVectorizer`` with ``norm=None`` or a ``CountVectorizer``
  pipeline variant depending on experimentation.

- ``sklearn.svm.LinearSVC`` — strong linear separator; use ``CalibratedClassifierCV``
  if you need ``predict_proba`` for top-k evaluation to match ``evaluate.py``.

Suggested workflow:
1. Reuse the same ``data/processed/*.csv`` splits as the baseline.
2. Share evaluation helpers from ``src.evaluate`` (``evaluate_model``, etc.).
3. Write metrics to ``results/`` with distinct filenames (e.g. ``metrics_nb.json``).
4. Keep CLI entrypoints small: ``python -m src.train_models`` can dispatch via
   subcommands or a ``--model`` flag when you add concrete training code.

This file intentionally stays minimal so the baseline path remains the
default for grading and reproducibility.
"""

from __future__ import annotations


def main() -> None:
    print(
        "train_models: placeholder. Add MultinomialNB / LinearSVC training here "
        "(see module docstring)."
    )


if __name__ == "__main__":
    main()
