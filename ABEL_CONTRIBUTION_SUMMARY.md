# Extended models: Naive Bayes and Linear SVM

## Overview

RouteRight compares the TF–IDF + logistic regression baseline to two additional
linear text classifiers trained on **`data/processed/train.csv`** with tuning
guided by **`val.csv`** and final numbers on **`test.csv`**.

## Models

### Multinomial Naive Bayes

- **Pipeline:** `CountVectorizer` (unigrams + bigrams, `min_df` in \{1, 2\}) + `MultinomialNB`.
- **Tuning:** `GridSearchCV` with **macro F1**; smoothing grid `alpha` in
  \{0.01, 0.1, 0.5, 1.0, 2.0, 5.0\}.
- **Outputs:** `results/multinomial_nb.joblib`, `results/metrics_nb.json`.

### Linear SVM with calibration

- **Pipeline:** `TfidfVectorizer` + `CalibratedClassifierCV(LinearSVC(...))`.
- **Tuning:** `C` grid on the inner `LinearSVC`, `min_df` on the vectorizer (same `{1,2}` idea as NB path).
- **Outputs:** `results/linear_svm_calibrated.joblib`, `results/metrics_svm.json`.

## Commands

```bash
python -m src.train_models --model nb
python -m src.train_models --model svm
python -m src.train_models --model all
```

Ensure splits exist (`python -m src.split_data --input data/processed/final_dataset.csv`) before training.

## Demo

```bash
python -m src.demo --model results/multinomial_nb.joblib "How do I reset my NetID password?"
python -m src.demo --model results/linear_svm_calibrated.joblib "..."
```

## Headline test metrics (representative full-corpus run)

Figures vary slightly if splits or hyperparameter search change; authoritative
numbers are always in **`results/metrics_nb.json`** and **`results/metrics_svm.json`**.
One consistent snapshot on the shared ~3k-example Yale crawl corpus:

| Model | Test accuracy | Test macro F1 | Test top‑3 accuracy |
|-------|---------------|----------------|---------------------|
| Multinomial NB | 0.864 | 0.842 | 0.960 |
| Calibrated Linear SVM | 0.861 | 0.838 | 0.960 |
| TF–IDF logistic (baseline) | 0.741 | 0.699 | 0.943 |

## Notes

- Training accuracy can be near-perfect while macro F1 on **Health_Wellness** and
  other minority offices remains harder; rely on confusion matrices and macro F1, not accuracy alone.
- Re-run **`python -m src.train_models`** after any change to `collected_questions.csv` → `clean_dataset` → `split_data` so metrics files stay aligned with `final_report.md`.
