# RouteRight: Yale Student-Support Intent Routing (Prototype)

**Team:** Ronald Milgo, Abel Adugna, Kevin Rusagara

## Introduction

- **Problem:** Yale students ask many short questions (email, forms, chatbots) that are easy to misroute to the wrong office, delaying answers and creating rework.
- **Motivation:** Turn unstructured, Yale-relevant questions into an actionable routing decision using supervised learning on **public Yale support and FAQ text** where possible.
- **Approach:** Multiclass text classification with TF-IDF features and a linear baseline, plus planned comparisons (Naive Bayes, Linear SVM).
- **Challenges:** Ambiguous multi-office wording, class imbalance between offices, and wording mismatch between formal FAQ lines and informal student phrasing.

## Data

- **Primary source:** Questions and headings scraped or copied with attribution from **public Yale** support and FAQ pages listed in `data/raw/faq_sources.csv` with canonical URLs on `yale.edu` and official Yale subdomains such as `yalehealth.yale.edu`.
- **Supplemental source (if needed):** If Yale only coverage is too small clearly mark rows in `collected_questions.csv` with `school` not equal to `Yale` and cite the non Yale URL. Summarize counts and domains in the report.
- **Labeling:** Follow `reports/labeling_rules.md` six fixed categories.
- **Cleaning:** `src/clean_dataset.py` produces `data/processed/final_dataset.csv` deduplicated validated rows with provenance columns for the report.
- **Splits:** Stratified train validation test for honest evaluation and a held out test set for final numbers.

## Methodology

- Learning formulation: supervised multiclass classification.
- Features: TF-IDF (unigrams and bigrams).
- Models: Logistic Regression baseline; planned Naive Bayes and Linear SVM comparisons; optional lightweight extensions if time permits.
- Metrics: accuracy, macro F1, weighted F1, per-class F1, top-3 accuracy.

## Implementation Details

- Repository layout: raw collection files, processed artifacts, results, models.
- Preprocessing: `clean_text` in `src/preprocess.py` plus dataset-level rules in `clean_dataset.py`.
- Hyperparameters: vectorizer n-grams, regularization, `max_iter` for logistic regression—document exact values used in the final run.

## Results

- Report validation and test metrics against proposal baselines (majority class and simple keyword rules) when implemented.
- Confusion matrix and per-class F1, with emphasis on smaller offices.
- Top-3 accuracy as a routing usefulness metric.

## Error Analysis

- Ambiguous queries and multi-intent examples.
- Yale versus supplemental wording if supplemental rows are used.

## Conclusion

- Summary, limitations, ethical use (no PII), and that deployment is out of scope for the course; future work on live ticketing integration.
