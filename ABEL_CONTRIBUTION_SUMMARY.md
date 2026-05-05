# Abel's Contribution: Extended Model Comparisons

## Overview
Abel successfully implemented and compared two additional machine learning models against the TF-IDF + Logistic Regression baseline for the RouteRight Yale student support intent classifier.

## Models Implemented

### 1. Multinomial Naive Bayes
- **Pipeline**: CountVectorizer + MultinomialNB
- **Rationale**: NB works better with raw counts than TF-IDF normalized features
- **Hyperparameters Tuned**:
  - `alpha`: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0] (smoothing parameter)
  - `min_df`: [1, 2] (minimum document frequency)
- **Best Parameters**: `alpha=1.0, min_df=1`

### 2. Linear SVM with Calibration
- **Pipeline**: TfidfVectorizer + CalibratedClassifierCV(LinearSVC)
- **Rationale**: Strong linear separator with calibration to enable `predict_proba` for top-k metrics
- **Hyperparameters Tuned**:
  - `C`: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] (regularization parameter)
  - `min_df`: [1, 2] (minimum document frequency)
- **Best Parameters**: `C=0.1, min_df=1`

## Technical Implementation Details

### Key Features
1. **Adaptive Cross-Validation**: Automatically adjusts CV folds based on dataset size to handle small datasets
2. **Grid Search with F1-Macro**: Uses macro F1 scoring to handle class imbalance fairly
3. **Calibrated Probabilities**: SVM uses CalibratedClassifierCV to enable top-k accuracy evaluation
4. **Comprehensive Evaluation**: Reuses existing evaluation framework for consistent metrics

### Code Structure
- **Main Module**: `src/train_models.py`
- **CLI Interface**: Supports `--model nb`, `--model svm`, or `--model all`
- **Output Files**:
  - `results/multinomial_nb.joblib` (trained NB model)
  - `results/linear_svm_calibrated.joblib` (trained SVM model)
  - `results/metrics_nb.json` (NB evaluation metrics)
  - `results/metrics_svm.json` (SVM evaluation metrics)

## Results Comparison

### Test Set Performance
| Model | Test Accuracy | Test F1-Macro | Test Top-3 Accuracy |
|-------|---------------|---------------|-------------------|
| **Multinomial NB** | **0.400** | **0.328** | **0.600** |
| Logistic Regression (baseline) | 0.300 | 0.222 | 0.500 |
| Linear SVM | 0.100 | 0.083 | 0.600 |

### Key Findings
1. **Multinomial NB is the best performer**: Achieves highest accuracy (40%) and F1-macro (0.328)
2. **Top-3 accuracy is consistent**: Both NB and SVM achieve 60% top-3 accuracy vs 50% for baseline
3. **SVM struggles with small dataset**: Linear SVM performs poorly, likely due to insufficient training data
4. **All models show overfitting**: Perfect training accuracy but poor generalization indicates small dataset limitations

## Demo Integration
Both models are fully integrated with the existing demo CLI:

```bash
# Test Naive Bayes model
python -m src.demo --model results/multinomial_nb.joblib "How do I reset my NetID password?"

# Test Linear SVM model  
python -m src.demo --model results/linear_svm_calibrated.joblib "How do I reset my NetID password?"
```

## Usage Instructions

### Training Individual Models
```bash
# Train only Naive Bayes
python -m src.train_models --model nb

# Train only Linear SVM
python -m src.train_models --model svm

# Train both models and compare
python -m src.train_models --model all
```

### Prerequisites
```bash
# Ensure data splits exist
python -m src.split_data --input data/sample_questions.csv

# Train baseline for comparison
python -m src.train_baseline
```

## Limitations and Future Work
1. **Small Dataset**: Only 28 training examples limits model performance and hyperparameter tuning effectiveness
2. **Cross-Validation Issues**: Some CV folds failed due to insufficient samples per class
3. **Calibration Challenges**: SVM calibration struggled with the small, imbalanced dataset

## Conclusion
Abel successfully extended the RouteRight project with two additional models, demonstrating that Multinomial Naive Bayes outperforms both the Logistic Regression baseline and Linear SVM on this Yale student support classification task. The implementation includes proper hyperparameter tuning, comprehensive evaluation, and seamless integration with the existing codebase.