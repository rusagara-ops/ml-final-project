"""Interactive demo: route a Yale student question to a support category.

Examples
--------
Single question (positional):
    python -m src.demo "How do I reset my NetID password?"

Read from stdin (e.g. piped or heredoc):
    echo "Where do I order an official transcript?" | python -m src.demo --stdin

Interactive REPL (one question per line, blank line or Ctrl-D to exit):
    python -m src.demo --interactive

Show top-5 instead of default top-3, and use a different model:
    python -m src.demo --top-k 5 --model results/some_other_model.joblib "I lost my Yale ID card"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from src.preprocess import clean_text

DEFAULT_MODEL_PATH = Path("results/logistic_regression_tfidf.joblib")
DEFAULT_TOP_K = 3
BAR_WIDTH = 24


def _format_confidence_bar(score: float, width: int = BAR_WIDTH) -> str:
    """Return a textual bar like ``[#########---------------]`` for a score in [0, 1]."""
    filled = max(0, min(width, int(round(score * width))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def predict(model, question: str, top_k: int) -> tuple[str, list[tuple[str, float]]]:
    """Return (top1, [(label, prob), ...]) for one question."""
    if not hasattr(model, "predict_proba"):
        raise SystemExit(
            "Loaded model has no predict_proba; demo needs probabilities. "
            "Wrap LinearSVC with CalibratedClassifierCV when training."
        )
    cleaned = clean_text(question)
    proba = model.predict_proba([cleaned])[0]
    classes = np.asarray(model.classes_)
    order = np.argsort(-proba)
    top_k = min(top_k, len(classes))
    top = [(str(classes[i]), float(proba[i])) for i in order[:top_k]]
    return str(classes[order[0]]), top


def render_prediction(question: str, top1: str, topk: list[tuple[str, float]]) -> str:
    lines = [
        f"Question:        {question}",
        f"Top-1 category:  {top1}",
        f"Top-{len(topk)} predictions:",
    ]
    for rank, (label, score) in enumerate(topk, start=1):
        bar = _format_confidence_bar(score)
        lines.append(f"  {rank}. {label:<16} {bar} {score * 100:5.1f}%")
    return "\n".join(lines)


def _iter_questions(args: argparse.Namespace) -> Iterable[str]:
    """Yield question strings based on parsed CLI arguments."""
    if args.interactive:
        try:
            while True:
                line = input("question> ").strip()
                if not line:
                    return
                yield line
        except (EOFError, KeyboardInterrupt):
            print()
            return
        return

    if args.stdin:
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield line
        return

    if args.question:
        yield " ".join(args.question).strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict Yale support category for a question.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Student question text (positional). Omit when using --stdin or --interactive.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to a joblib-pickled sklearn pipeline with predict_proba.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top predictions to display.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read one question per line from standard input.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt repeatedly; blank line or Ctrl-D to exit.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.stdin and args.interactive:
        parser.error("--stdin and --interactive are mutually exclusive")
    if not args.stdin and not args.interactive and not args.question:
        parser.error(
            "provide a question, or use --stdin / --interactive"
        )

    if not args.model.exists():
        raise SystemExit(
            f"Missing model at {args.model}. Train first: python -m src.train_baseline"
        )
    model = joblib.load(args.model)

    first = True
    for question in _iter_questions(args):
        if not question:
            continue
        top1, topk = predict(model, question, args.top_k)
        if not first:
            print()
        print(render_prediction(question, top1, topk))
        first = False


if __name__ == "__main__":
    main()
