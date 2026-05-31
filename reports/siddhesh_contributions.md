# Individual Contributions — Siddhesh Dayanand Patil
**DSC 445 — Machine Learning 1**  
**Branch:** `siddhesh`  
**DePaul University — MS in Artificial Intelligence**

---

## Overview

I was responsible for the complete data preprocessing pipeline, exploratory data analysis, and all machine learning modeling tasks including classification, regression, clustering, and model evaluation. My work spans six notebooks covering the full ML pipeline from raw data to final model comparison.

---

## Notebooks Completed

| Notebook | Description |
|----------|-------------|
| `sid_preprocessing.ipynb` | Data cleaning, merging, and bug fixes |
| `sid_eda.ipynb` | Exploratory data analysis and feature engineering |
| `sid_classification.ipynb` | Binary classification models |
| `sid_regression_models.ipynb` | Regression models for vote share prediction |
| `sid_clustering.ipynb` | Unsupervised county clustering |
| `sid_model_evaluation.ipynb` | Comprehensive model comparison and evaluation |

---

## 1. Data Preprocessing

### Pipeline
- Loaded 7 ACS Census tables (age, race, education, income, poverty, employment, housing) and MIT Election Lab presidential vote data
- Applied a shared `clean_acs()` function across all ACS tables for consistent cleaning
- Merged all tables on `county_fips` using inner joins
- Filtered MIT data to 2020 presidential election results
- Created two prediction targets: `party_winner` (binary) and `dem_vote_share` (continuous)

### Bug Fix 1 — ACS Sentinel Value
The U.S. Census Bureau uses `-666666666` as a placeholder for suppressed county-level data. Without replacing this value, all statistical computations on affected columns produce completely incorrect results. I identified and replaced all occurrences with `NaN` before merging.

### Bug Fix 2 — MIT Election Mode Filter
The MIT dataset reports votes broken down by mode: `TOTAL`, `ABSENTEE`, `EARLY VOTING`, `PROVISIONAL`, and others. The reference implementation filtered only `TOTAL` rows, silently dropping 849 counties that exclusively report by individual modes.

**Fix:** Use `TOTAL` rows where available; for counties without a `TOTAL` row, aggregate all modes.

| Approach | Counties Retained |
|----------|------------------|
| TOTAL only (reference) | 2,305 |
| Fixed approach | 3,154 |
| Final merged dataset | 3,115 |

### Final Dataset
- **Shape:** 3,115 counties × 162 features
- **Missing values:** 1 (median income, 1 county)
- **Republican-winning counties:** 2,576 (82.7%)
- **Democrat-winning counties:** 539 (17.3%)
- **States covered:** 51 (including DC)

---

## 2. Exploratory Data Analysis

### Class Imbalance Analysis
Identified a 4.78x class imbalance between Republican and Democrat counties. Documented why accuracy alone is not a reliable metric and recommended F1-score, Precision, Recall, and Confusion Matrix for evaluation.

### Swing County Analysis
Categorized all counties into three competitiveness buckets:

| Category | Vote Share | Counties | Share |
|----------|------------|----------|-------|
| Solid Republican | < 40% | 2,204 | 70.8% |
| Swing | 40%–60% | 673 | 21.6% |
| Solid Democrat | > 60% | 238 | 7.6% |

### Feature Engineering
Created 8 normalized rate/percentage features from raw ACS counts to enable fair cross-county comparison:

| Feature | Formula |
|---------|---------|
| `higher_ed_rate` | (Bachelor + Master + Prof + PhD) / Population |
| `poverty_rate` | Below poverty / Population |
| `unemployment_rate` | Unemployed / Labor force |
| `white_pct` | White population / Total population |
| `black_pct` | Black population / Total population |
| `asian_pct` | Asian population / Total population |
| `log_population` | log(1 + total population) |
| `housing_density` | Housing units / Population |

### Mutual Information Ranking
Computed mutual information scores to rank features by predictive power for `party_winner`:

| Rank | Feature | MI Score |
|------|---------|----------|
| 1 | white_pct | 0.1124 |
| 2 | asian_pct | 0.1101 |
| 3 | higher_ed_rate | 0.1048 |
| 4 | log_population | 0.0868 |
| 5 | black_pct | 0.0730 |
| 6 | median_income | 0.0399 |
| 7 | poverty_rate | 0.0098 |
| 8 | unemployment_rate | 0.0094 |
| 9 | housing_density | 0.0076 |

### Multicollinearity Findings
- `white_pct` vs `black_pct`: -0.81 — avoid using both in linear models
- `poverty_rate` vs `median_income`: -0.72 — avoid using both together
- `white_pct` vs `dem_vote_share`: -0.58 — strongest single linear predictor

### State-Level Analysis
Aggregated county data to state level revealing:
- Most Democrat-leaning: DC (0.921), Massachusetts (0.664), Hawaii (0.648)
- Most Republican-leaning: Nebraska (0.199), Oklahoma (0.203), West Virginia (0.243)
- Oklahoma and West Virginia: 0% Democrat-winning counties

---

## 3. Classification Models

**Target:** `party_winner` (0 = Republican, 1 = Democrat)  
**Split:** 80/20 stratified, `random_state=42`  
**Imbalance handling:** `class_weight='balanced'` / `scale_pos_weight`

### Models Trained
- Logistic Regression
- Random Forest (200 trees)
- XGBoost (200 estimators, scale_pos_weight=4.78)

### Results

| Model | Accuracy | Dem F1 | Dem Precision | Dem Recall | ROC AUC | CV F1 |
|-------|----------|--------|---------------|------------|---------|-------|
| Logistic Regression | 0.883 | 0.727 | 0.610 | 0.898 | 0.952 | 0.687 |
| **Random Forest** | **0.929** | **0.788** | **0.820** | 0.759 | 0.960 | 0.749 |
| XGBoost | 0.917 | 0.768 | 0.741 | 0.796 | **0.961** | **0.758** |

**Best model: Random Forest** — highest accuracy, F1, and precision on test set.

### Feature Importance (Random Forest)
Top features: `higher_ed_rate` (0.230), `white_pct` (0.198), `log_population` (0.127), `asian_pct` (0.116)

---

## 4. Regression Models

**Target:** `dem_vote_share` (continuous 0–1)  
**Split:** 80/20, `random_state=42`

### Models Trained
- Linear Regression
- Ridge (alpha=1.0)
- Lasso (alpha=0.001)
- Random Forest Regressor (200 trees)

### Results

| Model | R² | RMSE | MAE | CV R² |
|-------|-----|------|-----|-------|
| Linear Regression | 0.669 | 0.094 | 0.074 | 0.673 |
| Ridge | 0.669 | 0.094 | 0.074 | 0.673 |
| Lasso | 0.671 | 0.094 | 0.074 | 0.672 |
| **Random Forest** | **0.737** | **0.084** | **0.063** | **0.743** |

**Best model: Random Forest** — R² 0.737, predictions off by ~8.4% on average.

Key insight: Linear, Ridge, and Lasso all identical — regularization provides no benefit, confirming no overfitting with linear models on this dataset.

---

## 5. Clustering

**Algorithm:** K-Means  
**Features:** All 9 engineered features (no election labels used)

### Optimal k Selection

| k | Silhouette Score |
|---|-----------------|
| 2 | 0.3015 |
| **3** | **0.3094 (optimal)** |
| 4 | 0.1968 |
| 5+ | < 0.21 |

### Cluster Profiles

| Cluster | Profile | Avg Dem Share |
|---------|---------|---------------|
| 0 | Urban/Diverse — high education, high Asian% | 0.50 |
| 1 | Mixed/Swing — high Black%, moderate income | 0.42 |
| 2 | Rural/White — high white%, low education | 0.26 |

PCA visualization with 2 components explained 63.5% of variance with clear cluster separation.

**Key finding:** K-Means discovered politically-aligned county groups without ever seeing election data — demographics alone are sufficient to separate counties by voting behavior.

---

## 6. Model Evaluation Summary

| Task | Best Model | Key Metric |
|------|-----------|------------|
| Classification | Random Forest | F1: 0.788, AUC: 0.960 |
| Regression | Random Forest | R²: 0.737, RMSE: 0.084 |
| Clustering | K-Means (k=3) | Silhouette: 0.309 |

---

## Key Findings and Insights

1. **Race and education dominate** — white_pct, higher_ed_rate, and asian_pct are the top 3 predictors consistently across EDA, classification, and regression
2. **Demographics explain 73.7% of vote share variance** — a remarkably strong signal from Census data alone
3. **Poverty and unemployment are weak predictors** — despite being economically important, they barely predict voting behavior on their own
4. **4.78x class imbalance requires explicit handling** — naive accuracy would be 82.7% by always predicting Republican
5. **Random Forest outperforms linear models by +0.068 R²** — non-linear demographic interactions matter
6. **Clustering validates EDA** — unsupervised discovery of 3 clusters matches political intuition perfectly

---

## Challenges and How They Were Resolved

| Challenge | Resolution |
|-----------|------------|
| ACS sentinel value -666666666 causing incorrect statistics | Identified and replaced all occurrences with NaN |
| MIT mode filter silently dropping 849 counties | Fixed by using TOTAL where available, aggregating modes otherwise |
| NAME column ending in 'E' being misidentified as estimate column | Added explicit skip set in clean_acs() function |
| 4.78x class imbalance causing biased models | Used class_weight='balanced' for all classification models |
| Raw counts biased by county size | Engineered normalized rate features for all demographic variables |
| PerformanceWarning on fragmented DataFrame | Used df.copy() before adding engineered features |

---

## Conclusions and Recommendations

- **Use Random Forest** as the primary model for both classification and regression tasks
- **Avoid using white_pct and black_pct together** in linear models — correlation of -0.81 causes multicollinearity
- **Evaluate on swing counties separately** — the 673 competitive counties (40-60% vote share) are the hardest to predict and most valuable
- **Do not rely on accuracy** — use F1-score, Precision, Recall, and AUC as primary metrics
- **For future work:** incorporating historical election data (2012, 2016) as trend features would likely improve model performance significantly

---

*Siddhesh Dayanand Patil — DSC 445 Machine Learning 1 — DePaul University — May 2026*
