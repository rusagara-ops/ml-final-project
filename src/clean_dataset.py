"""Clean and validate raw collected questions into a training-ready dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.preprocess import clean_text

ALLOWED_LABELS = frozenset(
    {
        "Registrar",
        "Financial_Aid",
        "Housing",
        "IT",
        "Dining",
        "Health_Wellness",
    }
)


def clean_dataset(
    df: pd.DataFrame,
    *,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return cleaned dataframe and counts of dropped/skipped rows."""
    stats = {
        "input_rows": int(len(df)),
        "empty_after_clean": 0,
        "invalid_label": 0,
        "duplicate_clean_text": 0,
        "output_rows": 0,
    }

    required = {"text", "label", "source_name", "school", "url"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work["_clean"] = work["text"].map(clean_text)
    before = len(work)
    work = work[work["_clean"].str.len() > 0]
    stats["empty_after_clean"] = before - len(work)

    valid_mask = work["label"].astype(str).isin(ALLOWED_LABELS)
    invalid = int((~valid_mask).sum())
    if strict and invalid:
        bad = work.loc[~valid_mask, "label"].drop_duplicates().tolist()
        raise ValueError(f"Invalid labels present (strict mode): {bad}")
    work = work.loc[valid_mask]
    stats["invalid_label"] = invalid

    before_dedupe = len(work)
    work = work.drop_duplicates(subset=["_clean"], keep="first")
    stats["duplicate_clean_text"] = before_dedupe - len(work)

    out = work.assign(text=work["_clean"]).drop(columns=["_clean"])
    # Canonical column order for downstream + provenance
    cols = ["text", "label", "source_name", "school", "url"]
    out = out[cols]
    stats["output_rows"] = int(len(out))
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean collected_questions.csv into processed/final_dataset.csv"
    )
    parser.add_argument(
        "--input",
        default="data/raw/collected_questions.csv",
        help="Raw CSV with columns: text,label,source_name,school,url",
    )
    parser.add_argument(
        "--output",
        default="data/processed/final_dataset.csv",
        help="Output CSV path (parent dirs created as needed)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row has a label outside the 6 allowed categories",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)
    cleaned, stats = clean_dataset(df, strict=args.strict)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({stats['output_rows']} rows)")
    print(
        "Stats:",
        {k: v for k, v in stats.items() if k != "output_rows"},
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
