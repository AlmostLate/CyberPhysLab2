# ML Results - Lab 2 NLP

## Experiment Configuration

This document contains the results of traditional ML model experiments for credit scoring.

## Dataset
- UCI Adult Income Dataset
- Task: Binary classification (income >50K or <=50K)
- Features: 14 demographic and employment features

## Models Tested

### 1. Logistic Regression
- **Type**: Linear classifier
- **Parameters**: C=1.0, max_iter=1000, solver=lbfgs
- **Pros**: Interpretable, probabilistic output
- **Cons**: Limited to linear relationships

### 2. Random Forest
- **Type**: Ensemble (bagging)
- **Parameters**: n_estimators=100, max_depth=10, min_samples_split=5
- **Pros**: Handles non-linear relationships, feature importance
- **Cons**: Less interpretable than logistic regression

### 3. Gradient Boosting
- **Type**: Ensemble (boosting)
- **Parameters**: n_estimators=100, learning_rate=0.1, max_depth=5
- **Pros**: High accuracy, handles complex patterns
- **Cons**: Risk of overfitting, slower training

## Results (TBD - Requires Training)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD | TBD |

## Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | TBD | TBD |
| 2 | TBD | TBD |
| 3 | TBD | TBD |

## Conclusions

Traditional ML models provide:
- Baseline performance for comparison with LLM-based approaches
- Interpretable results (especially logistic regression)
- Feature importance insights for credit scoring factors
