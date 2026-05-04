"""Generate report figures from a trained pipeline + processed splits.

Produces, by default:
- ``reports/figures/class_distribution.png``    train/val/test counts per label
- ``reports/figures/confusion_matrix_test.png`` row-normalized confusion matrix
- ``reports/figures/per_class_metrics.png``     precision/recall/F1 bars per class
- ``reports/figures/topk_accuracy.png``         top-k accuracy curve k = 1..K
- ``reports/figures/confidence_distribution.png`` correct-vs-incorrect confidence
- ``reports/figures/top_features_per_class.png`` most positive TF-IDF features

Each figure is saved with ``bbox_inches="tight"``. Re-run after retraining.

Usage
-----
    python -m src.figures
    python -m src.figures --model results/logistic_regression_tfidf.joblib \
                          --split test --out-dir reports/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

PROCESSED_DIR = Path("data/processed")
DEFAULT_MODEL = Path("results/logistic_regression_tfidf.joblib")
DEFAULT_OUT_DIR = Path("reports/figures")
SPLITS = ("train", "val", "test")


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.split_data --input data/processed/final_dataset.csv"
        )
    return pd.read_csv(path)


def plot_class_distribution(out_path: Path) -> None:
    counts = {}
    for split in SPLITS:
        df = _load_split(split)
        counts[split] = df["label"].value_counts()
    classes = sorted({c for s in counts.values() for c in s.index})
    width = 0.27
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, split in enumerate(SPLITS):
        values = [int(counts[split].get(c, 0)) for c in classes]
        ax.bar(x + (i - 1) * width, values, width=width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("Number of questions")
    ax.set_title("Class distribution across splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(model, df: pd.DataFrame, out_path: Path, split_name: str) -> np.ndarray:
    X = df["text"].astype(str)
    y_true = df["label"].astype(str).to_numpy()
    y_pred = model.predict(X)

    classes = np.asarray(model.classes_)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix ({split_name}, row-normalized)")

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:d}\n{val * 100:.0f}%",
                    ha="center", va="center", fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized share")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_per_class_metrics(model, df: pd.DataFrame, out_path: Path) -> None:
    X = df["text"].astype(str)
    y_true = df["label"].astype(str).to_numpy()
    y_pred = model.predict(X)
    classes = np.asarray(model.classes_)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0,
    )

    width = 0.27
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width, precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={int(s)})" for c, s in zip(classes, support)],
                       rotation=15, ha="center")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1 (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_topk_accuracy(model, df: pd.DataFrame, out_path: Path, max_k: int = 6) -> None:
    if not hasattr(model, "predict_proba"):
        return
    X = df["text"].astype(str)
    y_true = df["label"].astype(str).to_numpy()
    classes = np.asarray(model.classes_)
    proba = model.predict_proba(X)
    order = np.argsort(-proba, axis=1)
    ranked = classes[order]

    ks = np.arange(1, min(max_k, len(classes)) + 1)
    accs = []
    for k in ks:
        accs.append(float(np.mean(np.any(ranked[:, :k] == y_true[:, None], axis=1))))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, accs, marker="o", color="tab:blue")
    for k, a in zip(ks, accs):
        ax.annotate(f"{a:.3f}", (k, a), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Top-k accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Top-k routing accuracy on test set")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_distribution(model, df: pd.DataFrame, out_path: Path) -> None:
    if not hasattr(model, "predict_proba"):
        return
    X = df["text"].astype(str)
    y_true = df["label"].astype(str).to_numpy()
    classes = np.asarray(model.classes_)
    proba = model.predict_proba(X)
    top_idx = proba.argmax(axis=1)
    top_score = proba[np.arange(len(proba)), top_idx]
    pred = classes[top_idx]
    correct = pred == y_true

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 21)
    ax.hist(top_score[correct], bins=bins, alpha=0.7,
            label=f"correct (n={int(correct.sum())})", color="tab:green")
    ax.hist(top_score[~correct], bins=bins, alpha=0.7,
            label=f"incorrect (n={int((~correct).sum())})", color="tab:red")
    ax.set_xlabel("Top-1 predicted probability")
    ax.set_ylabel("Number of test questions")
    ax.set_title("Confidence distribution on test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_top_features_per_class(model, out_path: Path, top_n: int = 10) -> None:
    """Show top-n most positive TF-IDF features for each class (one-vs-rest weights)."""
    try:
        vectorizer = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
    except (AttributeError, KeyError):
        return
    if not hasattr(clf, "coef_"):
        return

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    classes = np.asarray(clf.classes_)
    n_cls = len(classes)
    cols = 2
    rows = (n_cls + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows + 0.5))
    axes = axes.ravel()
    for idx, cls in enumerate(classes):
        coefs = clf.coef_[idx]
        top_idx = np.argsort(-coefs)[:top_n]
        feats = feature_names[top_idx][::-1]
        weights = coefs[top_idx][::-1]

        ax = axes[idx]
        ax.barh(np.arange(len(feats)), weights, color="tab:blue")
        ax.set_yticks(np.arange(len(feats)))
        ax.set_yticklabels(feats, fontsize=9)
        ax.set_title(str(cls), fontsize=11)
        ax.set_xlabel("Logistic regression weight")
    for k in range(n_cls, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Top {top_n} TF-IDF features per class", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate report figures from a trained pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to a joblib-pickled sklearn pipeline.")
    parser.add_argument("--split", choices=SPLITS, default="test",
                        help="Split to evaluate confusion matrix and per-class metrics on.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory to write figures into.")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Missing model at {args.model}. Train first: python -m src.train_baseline"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)
    eval_df = _load_split(args.split)

    plot_class_distribution(args.out_dir / "class_distribution.png")
    plot_confusion_matrix(model, eval_df, args.out_dir / f"confusion_matrix_{args.split}.png", args.split)
    plot_per_class_metrics(model, eval_df, args.out_dir / "per_class_metrics.png")
    plot_topk_accuracy(model, eval_df, args.out_dir / "topk_accuracy.png")
    plot_confidence_distribution(model, eval_df, args.out_dir / "confidence_distribution.png")
    plot_top_features_per_class(model, args.out_dir / "top_features_per_class.png")

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
