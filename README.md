# RouteRight: Yale Student-Support Intent Classifier (Prototype)

**Course:** CPSC 381/581 Machine Learning — Final Project  
**Team:** Ronald Milgo, Abel Adugna, Kevin Rusagara

## Overview

**RouteRight** is a **Yale-focused** machine learning prototype that routes short student questions to Yale relevant support categories. The baseline uses **TF-IDF** features (unigrams and bigrams) and **Logistic Regression** (`scikit-learn`). Training data should come primarily from **public Yale support and FAQ pages**; see [Real Dataset Collection](#real-dataset-collection) below.

## Problem statement

Yale students submit many questions through email forms and helpdesk style channels. Misrouting delays answers and adds work for staff. This project learns to map a short question to one of six support categories so routing is faster and more consistent than ad hoc keyword rules alone.

## Categories

| Label | Yale-relevant routing (examples) |
|--------|-----------------------------------|
| `Registrar` | Transcripts enrollment verification registration add drop degree records |
| `Financial_Aid` | FAFSA Yale aid scholarships loans work study verification |
| `Housing` | Yale Housing portals room draw roommate moves dorm policies |
| `IT` | NetID MFA email Wi‑Fi VPN **Canvas @ Yale** and campus IT guides |
| `Dining` | Yale Hospitality meal plans dining halls menus dietary accommodations |
| `Health_Wellness` | Yale Health student coverage immunizations counseling accessibility tied to health services |

Detailed labeling rules: `reports/labeling_rules.md`.

## Repository structure

```text
ml-final-project/
├── data/
│   ├── raw/
│   │   ├── faq_sources.csv                     # high-level Yale source list (by category)
│   │   ├── yale_faq_seed_urls.csv             # HTML seeds for the polite crawler
│   │   ├── collected_questions_yale_scraped.csv # auto-extracted lines (regenerate with script)
│   │   └── collected_questions.csv            # merged manual + crawl (canonical raw table)
│   ├── processed/                 # generated: final_dataset.csv train val test (gitignored)
│   └── sample_questions.csv       # small dev set for quick local runs
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── clean_dataset.py           # raw → validated processed/final_dataset.csv
│   ├── collect_yale_faqs.py       # polite crawl of Yale HTML FAQ pages → scraped CSV
│   ├── split_data.py
│   ├── train_baseline.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── demo.py
├── results/
├── reports/
│   ├── labeling_rules.md
│   └── final_report_outline.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Use Python 3.10+ (3.11 recommended). From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run module commands from the **repository root** so relative paths resolve. On macOS, if `python` is not on your PATH, use **`python3`** (and `python3 -m venv .venv`) the same way.

## Teammate quickstart (after `git clone`)

1. **Create a branch** off `main` for your work (`git checkout -b your-name/feature`).
2. **Install deps** (see [Setup](#setup)).
3. **Regenerate local artifacts** — `data/processed/` and `results/*.joblib` are **gitignored**, so a fresh clone does **not** include `train.csv` or the trained model. Run the **full Yale pipeline** once (uses tracked `data/raw/collected_questions.csv`):

   ```bash
   python -m src.clean_dataset
   python -m src.split_data --input data/processed/final_dataset.csv
   python -m src.train_baseline
   python -m src.demo "How do I reset my NetID password?"
   ```

   After that, **`src.demo`** and any training that reads `data/processed/train.csv` will work.

4. **Who touches what (high level)**  
   - **Ronald (landed):** repo layout, `preprocess` / `clean_dataset` / `collect_yale_faqs`, Yale raw tables, `split_data`, `train_baseline`, `evaluate`, baseline `demo`.  
   - **Abel:** extend **`src/train_models.py`** (NB, Linear SVM, tuning); reuse **`data/processed/*.csv`** splits and **`src/evaluate.py`**.  
   - **Kevin:** polish **`src/demo`**, confusion matrix / figures for the report, draft → PDF using **`reports/final_report_outline.md`**.

5. **Before opening a PR:** run the block in step 3 (or the [quick dev loop](#commands-quick-dev-loop) if you only changed code unrelated to the big dataset), fix any errors, and note in the PR what you changed (data vs models vs docs).

## Real Dataset Collection

- **Starter CSV (`data/sample_questions.csv`):** Generic examples for **development only** quick tests of the training pipeline. It is **not** the Yale production dataset.
- **Primary data goal:** Build `data/raw/collected_questions.csv` from **Yale public FAQ and support pages** (see `data/raw/faq_sources.csv`). Each row must include `text`, `label`, `source_name`, `school`, and `url` for reproducibility and the final report.
- **Target size:** About **2,000–5,000** labeled snippets after cleaning if feasible. The repo includes an **automated crawl** (below) that can reach that range from Yale HTML pages; **review or filter** noisy rows before treating the set as gold labels.
- **Cleaning:** Run `python -m src.clean_dataset` to produce **`data/processed/final_dataset.csv`** with cleaned text deduplication empty rows removed and labels restricted to the six allowed categories.
- **Final training path:** For coursework deliverables the **intended** training input is **`data/processed/final_dataset.csv`** once it is populated from Yale sources. Then split and train:

  ```bash
  python -m src.clean_dataset
  python -m src.split_data --input data/processed/final_dataset.csv
  python -m src.train_baseline
  ```

- **Supplemental non Yale FAQs:** Allowed **only** if Yale only material is too small. Mark those rows with a non `Yale` value in `school` and a non Yale `url` and describe totals in the report so reviewers can see what is Yale native versus supplemental.

- **Automated Yale crawl (scale-up):** `src/collect_yale_faqs.py` fetches **HTML only** from hostnames mapped to the six labels, honors `robots.txt`, uses a **slow default delay** between requests, and extracts FAQ-like headings and list items. Seeds live in **`data/raw/yale_faq_seed_urls.csv`** (add more Yale HTML URLs there to grow coverage). Example:

  ```bash
  python -m src.collect_yale_faqs --max-pages 350 --delay 1.25 --output data/raw/collected_questions_yale_scraped.csv
  ```

  To **append** crawl output into the hand-curated file (deduped on cleaned text):

  ```bash
  python -m src.collect_yale_faqs --max-pages 350 --delay 1.25 --merge-into data/raw/collected_questions.csv
  ```

  Then run **`clean_dataset`** again. **Do not** point high volume parallel scrapers at Yale; keep delays conservative and follow Yale’s terms of use. Prefer **manual spot checks** and deletion of bad lines (navigation boilerplate, non-question paragraphs) before final reporting.

## Commands (quick dev loop)

Uses the small **starter** file:

```bash
python -m src.split_data --input data/sample_questions.csv
python -m src.train_baseline
python -m src.demo "How do I request an official transcript?"
```

### Demo CLI options

`src/demo.py` supports a few input modes and a `--top-k` flag, plus
`--model` so it can also showcase Abel's models once they exist:

```bash
python -m src.demo "How do I drop a course?"                 # positional
python -m src.demo --top-k 5 "Where do I file a FAFSA appeal?"
python -m src.demo --interactive                              # REPL mode
echo "I lost my Yale ID card" | python -m src.demo --stdin
python -m src.demo --model results/some_other_model.joblib "..."
```

## Commands (Yale dataset workflow)

1. **Optional — expand Yale HTML data:** edit **`data/raw/yale_faq_seed_urls.csv`**, then run **`collect_yale_faqs`** (see [Real Dataset Collection](#real-dataset-collection)). Merge into **`data/raw/collected_questions.csv`** when ready.

2. **Manual curation:** edit **`data/raw/collected_questions.csv`** following **`reports/labeling_rules.md`**. Add or remove rows; mark any non-Yale supplemental rows with `school` and `url` accordingly.

3. **Clean and validate:**

   ```bash
   python -m src.clean_dataset
   ```

   Options:

   ```bash
   python -m src.clean_dataset --input data/raw/collected_questions.csv --output data/processed/final_dataset.csv
   python -m src.clean_dataset --strict   # fail if any label is invalid
   ```

4. **Split** then **train:**

   ```bash
   python -m src.split_data --input data/processed/final_dataset.csv
   python -m src.train_baseline
   ```

Optional placeholder for future model work:

```bash
python -m src.train_models
```

## Team workflow

| Member | Focus |
|--------|--------|
| **Ronald** | Repo setup Yale data workflow preprocessing baseline training |
| **Abel** | Model comparisons and tuning (`train_models.py` hyperparameters) |
| **Kevin** | Demo polish top three output confusion matrix final report PDF |

Keep `main` stable for submission. Use feature branches and short PRs so code stays mergeable and reproducible for the grader.

## Report

Align the PDF with `reports/final_report_outline.md` Introduction Data Methodology Implementation Details Results Error Analysis Conclusion. Cite Yale versus supplemental rows and include preprocessing and hyperparameters as required by the course spec.

## Artifacts

| Path | Role |
|------|------|
| `data/raw/faq_sources.csv`, `data/raw/yale_faq_seed_urls.csv` | Human-curated source lists (tracked in git). |
| `data/raw/collected_questions.csv`, `data/raw/collected_questions_yale_scraped.csv` | Raw labeled snippets + crawl output (**tracked** so teammates share the same ~3k Yale dataset). |
| `data/processed/final_dataset.csv` | Cleaned table from `clean_dataset` (**gitignored** — regenerate). |
| `data/processed/train.csv`, `val.csv`, `test.csv` | Splits (**gitignored** — regenerate). |
| `results/logistic_regression_tfidf.joblib` | Trained pipeline (**gitignored** — run `train_baseline`). |
| `results/metrics_baseline.json` | Metrics (**gitignored** — run `train_baseline`). |
| `results/.gitkeep` | Keeps the `results/` folder in git without storing large binaries. |

For **Canvas zip submission**, include a clean tree: follow course rules, ship **README + source**, and exclude **`.venv/`** unless the course says otherwise. Graders should be able to run the commands in [Teammate quickstart](#teammate-quickstart-after-git-clone) from a clone.
