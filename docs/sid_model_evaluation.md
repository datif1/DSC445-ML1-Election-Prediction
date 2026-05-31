# Sid — Model Evaluation

**Notebook:** `sid_model_evaluation.ipynb`  
**Branch:** `siddhesh`  
**Input:** `../data/processed/model_ready_dataset.csv`

---

## Overview

Comprehensive evaluation and comparison of all models across classification, regression, and clustering tasks using both test set metrics and 5-fold cross-validation.

---

## Classification Results

| Model | Accuracy | Dem F1 | Dem Precision | Dem Recall | ROC AUC | CV F1 |
|-------|----------|--------|---------------|------------|---------|-------|
| Logistic Regression | 0.883 | 0.727 | 0.610 | 0.898 | 0.952 | 0.687 |
| Random Forest | **0.929** | **0.788** | **0.820** | 0.759 | 0.960 | 0.749 |
| XGBoost | 0.917 | 0.768 | 0.741 | **0.796** | **0.961** | **0.758** |

**Best classifier: Random Forest** (test F1) / XGBoost (CV F1)

---

## Regression Results

| Model | R² | RMSE | MAE | CV R² |
|-------|-----|------|-----|-------|
| Linear Regression | 0.6688 | 0.0943 | 0.0739 | 0.673 |
| Ridge | 0.6688 | 0.0943 | 0.0738 | 0.673 |
| Lasso | 0.6708 | 0.0940 | 0.0736 | 0.672 |
| Random Forest | **0.7367** | **0.0841** | **0.0629** | **0.743** |

**Best regressor: Random Forest**

---

## Clustering Results

| Metric | Value |
|--------|-------|
| Algorithm | K-Means |
| Optimal k | 3 |
| Silhouette Score | 0.309 |
| PCA Variance Explained | 63.5% |
| Cluster 0 — Urban/Diverse | Avg dem share: 0.50 |
| Cluster 1 — Mixed/Swing | Avg dem share: 0.42 |
| Cluster 2 — Rural/White | Avg dem share: 0.26 |

---

## Key Findings

1. **Random Forest is the best model for both supervised tasks** — classification and regression
2. **Top features consistent across all tasks:** white_pct, higher_ed_rate, asian_pct
3. **Class imbalance (4.78x) handled via class_weight** — do not use accuracy alone
4. **Demographics explain 73.7% of vote share variance** — strong signal
5. **K-Means found political groups without election labels** — demographics drive voting behavior

---

## Recommendations for Team

- Use `class_weight='balanced'` for all classification models
- Avoid `white_pct` + `black_pct` together in linear models (correlation -0.81)
- Avoid `poverty_rate` + `median_income` together (correlation -0.72)
- Use `log_population` not raw population
- Evaluate on swing counties (673 counties, 40-60% dem share) separately
- Consider XGBoost for final model — highest CV F1 (0.758)
