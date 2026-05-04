"""Create stratified train/validation/test splits from labeled questions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import clean_text

RANDOM_STATE = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split labeled CSV into train/val/test (stratified)."
    )
    parser.add_argument(
        "--input",
        default="data/sample_questions.csv",
        help="Path to CSV with columns: text,label",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Directory for train.csv, val.csv, test.csv",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.6,
        help="Fraction for training (remaining split evenly val/test)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Input CSV must contain columns: text, label")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

    strat = df["label"] if df["label"].nunique() > 1 else None
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_size,
        random_state=RANDOM_STATE,
        stratify=strat,
    )
    strat_temp = temp_df["label"] if temp_df["label"].nunique() > 1 else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=strat_temp,
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"Wrote {len(train_df)} train, {len(val_df)} val, {len(test_df)} test rows to {out_dir}")


if __name__ == "__main__":
    main()
