"""Extended model comparisons: Multinomial Naive Bayes and calibrated Linear SVM.

Trains alternative linear text classifiers on the same stratified splits as the
TF–IDF + logistic regression baseline:
- MultinomialNB with CountVectorizer and smoothing (`alpha`) tuning
- LinearSVC inside CalibratedClassifierCV for probability outputs and top‑k metrics
- GridSearchCV with macro F1 scoring
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import evaluate_model

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def build_nb_pipeline() -> Pipeline:
    """Build Multinomial Naive Bayes pipeline with CountVectorizer.
    
    NB works better with raw counts than TF-IDF normalized features.
    """
    return Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "clf",
                MultinomialNB(),
            ),
        ]
    )


def build_svm_pipeline() -> Pipeline:
    """Build Linear SVM pipeline with CalibratedClassifierCV for probabilities.
    
    Uses TF-IDF features and calibration to enable predict_proba for top-k metrics.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(max_iter=5000, random_state=42),
                    cv=3,  # 3-fold CV for calibration
                ),
            ),
        ]
    )


def tune_naive_bayes(X_train, y_train, X_val, y_val) -> tuple[Pipeline, dict[str, Any]]:
    """Tune Multinomial NB hyperparameters using validation set."""
    print("Tuning Multinomial Naive Bayes...")
    
    # Define parameter grid
    param_grid = {
        'clf__alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],  # Smoothing parameter
        'vectorizer__min_df': [1, 2],  # Minimum document frequency
    }
    
    base_pipeline = build_nb_pipeline()
    
    # Use 2-fold CV for small datasets
    cv_folds = min(3, len(X_train) // 6)  # Ensure at least 6 samples per fold
    cv_folds = max(2, cv_folds)  # Minimum 2 folds
    
    # Grid search with reduced CV folds
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=cv_folds,
        scoring='f1_macro',  # Use macro F1 to handle class imbalance
        n_jobs=1,  # Reduce parallelism for small datasets
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Evaluate on validation set
    val_metrics = evaluate_model(best_model, X_val, y_val)
    
    tuning_info = {
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'validation_metrics': val_metrics
    }
    
    print(f"Best NB params: {grid_search.best_params_}")
    print(f"Best CV F1-macro: {grid_search.best_score_:.3f}")
    print(f"Validation F1-macro: {val_metrics['f1_macro']:.3f}")
    
    return best_model, tuning_info


def tune_linear_svm(X_train, y_train, X_val, y_val) -> tuple[Pipeline, dict[str, Any]]:
    """Tune Linear SVM hyperparameters using validation set."""
    print("Tuning Linear SVM with Calibration...")
    
    # Define parameter grid
    param_grid = {
        'clf__estimator__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # Regularization
        'tfidf__min_df': [1, 2],  # Minimum document frequency
    }
    
    base_pipeline = build_svm_pipeline()
    
    # Use 2-fold CV for small datasets and reduce calibration CV
    cv_folds = min(3, len(X_train) // 6)  # Ensure at least 6 samples per fold
    cv_folds = max(2, cv_folds)  # Minimum 2 folds
    
    # Modify the SVM pipeline to use fewer CV folds for calibration
    base_pipeline.named_steps['clf'].cv = 2
    
    # Grid search with reduced CV folds
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=cv_folds,
        scoring='f1_macro',  # Use macro F1 to handle class imbalance
        n_jobs=1,  # Reduce parallelism for small datasets
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Evaluate on validation set
    val_metrics = evaluate_model(best_model, X_val, y_val)
    
    tuning_info = {
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'validation_metrics': val_metrics
    }
    
    print(f"Best SVM params: {grid_search.best_params_}")
    print(f"Best CV F1-macro: {grid_search.best_score_:.3f}")
    print(f"Validation F1-macro: {val_metrics['f1_macro']:.3f}")
    
    return best_model, tuning_info


def train_and_evaluate_model(model_name: str, X_train, y_train, X_val, y_val, X_test, y_test) -> dict[str, Any]:
    """Train and evaluate a specific model."""
    print(f"\n=== Training {model_name} ===")
    
    if model_name == "nb":
        model, tuning_info = tune_naive_bayes(X_train, y_train, X_val, y_val)
        model_path = RESULTS_DIR / "multinomial_nb.joblib"
        metrics_path = RESULTS_DIR / "metrics_nb.json"
    elif model_name == "svm":
        model, tuning_info = tune_linear_svm(X_train, y_train, X_val, y_val)
        model_path = RESULTS_DIR / "linear_svm_calibrated.joblib"
        metrics_path = RESULTS_DIR / "metrics_svm.json"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Save the trained model
    joblib.dump(model, model_path)
    
    # Evaluate on all splits
    metrics = {
        "model_name": model_name,
        "model_path": str(model_path),
        "tuning_info": tuning_info,
        "train": evaluate_model(model, X_train, y_train),
        "validation": evaluate_model(model, X_val, y_val),
        "test": evaluate_model(model, X_test, y_test),
    }
    
    # Save metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults for {model_name.upper()}:")
    print(f"Train Accuracy: {metrics['train']['accuracy']:.3f}")
    print(f"Val Accuracy:   {metrics['validation']['accuracy']:.3f}")
    print(f"Test Accuracy:  {metrics['test']['accuracy']:.3f}")
    print(f"Test F1-macro:  {metrics['test']['f1_macro']:.3f}")
    print(f"Test Top-3:     {metrics['test']['top3_accuracy']:.3f}")
    
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    
    return metrics


def compare_all_models() -> None:
    """Train and compare all models (NB and SVM)."""
    # Load data
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"
    
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            "Missing data splits. Run: python -m src.split_data --input data/sample_questions.csv"
        )
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(str)
    X_val, y_val = val_df["text"].astype(str), val_df["label"].astype(str)
    X_test, y_test = test_df["text"].astype(str), test_df["label"].astype(str)
    
    print(f"Dataset sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Classes: {sorted(y_train.unique())}")
    
    # Train both models
    nb_metrics = train_and_evaluate_model("nb", X_train, y_train, X_val, y_val, X_test, y_test)
    svm_metrics = train_and_evaluate_model("svm", X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Print comparison summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Test Acc':<10} {'Test F1-macro':<15} {'Test Top-3':<10}")
    print("-"*60)
    
    for name, metrics in [("Multinomial NB", nb_metrics), ("Linear SVM", svm_metrics)]:
        acc = metrics['test']['accuracy']
        f1 = metrics['test']['f1_macro']
        top3 = metrics['test']['top3_accuracy']
        print(f"{name:<20} {acc:<10.3f} {f1:<15.3f} {top3:<10.3f}")
    
    # Load baseline for comparison if it exists
    baseline_path = RESULTS_DIR / "metrics_baseline.json"
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)
        if 'test' in baseline_metrics:
            acc = baseline_metrics['test']['accuracy']
            f1 = baseline_metrics['test']['f1_macro']
            top3 = baseline_metrics['test']['top3_accuracy']
            print(f"{'Logistic Reg (baseline)':<20} {acc:<10.3f} {f1:<15.3f} {top3:<10.3f}")


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train and compare Multinomial NB and Linear SVM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["nb", "svm", "all"],
        default="all",
        help="Which model to train: 'nb' (Naive Bayes), 'svm' (Linear SVM), or 'all'.",
    )
    
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.model == "all":
        compare_all_models()
    else:
        # Load data for single model training
        train_path = PROCESSED_DIR / "train.csv"
        val_path = PROCESSED_DIR / "val.csv"
        test_path = PROCESSED_DIR / "test.csv"
        
        if not all(p.exists() for p in [train_path, val_path, test_path]):
            raise FileNotFoundError(
                "Missing data splits. Run: python -m src.split_data --input data/sample_questions.csv"
            )
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(str)
        X_val, y_val = val_df["text"].astype(str), val_df["label"].astype(str)
        X_test, y_test = test_df["text"].astype(str), test_df["label"].astype(str)
        
        train_and_evaluate_model(args.model, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    main()
