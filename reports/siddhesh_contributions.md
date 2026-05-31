# Individual Contributions — Siddhesh Dayanand Patil
**DSC 445 — Machine Learning 1**  
**Branch:** `siddhesh`  
**DePaul University — MS in Artificial Intelligence**

---

## Overview

I was responsible for the complete data preprocessing pipeline, exploratory data analysis, baseline modeling, and an advanced historical modeling pipeline that predicts 2024 county-level presidential election outcomes using election data from 2000–2020 combined with ACS 2020 demographic features.

My work spans **9 notebooks** covering the full ML pipeline from raw data ingestion to comprehensive model evaluation.

---

## Notebooks Completed

| Notebook | Description |
|----------|-------------|
| `sid_preprocessing.ipynb` | Data cleaning, merging, bug fixes |
| `sid_eda.ipynb` | Exploratory data analysis and feature engineering |
| `sid_classification.ipynb` | Baseline classification (2020 demographics only) |
| `sid_regression_models.ipynb` | Baseline regression (2020 demographics only) |
| `sid_clustering.ipynb` | Baseline clustering (2020 demographics only) |
| `sid_model_evaluation.ipynb` | Baseline model comparison |
| `sid_historical_preprocessing.ipynb` | Historical feature engineering (2000–2020) |
| `sid_historical_classification.ipynb` | Historical classification — predicting 2024 |
| `sid_historical_regression.ipynb` | Historical regression — predicting 2024 vote share |
| `sid_historical_clustering.ipynb` | Historical clustering with full feature set |
| `sid_model_evaluation_v2.ipynb` | Full comparison: baseline vs historical |

---

## Part 1 — Data Preprocessing

### Pipeline
- Loaded 7 ACS Census tables and MIT Election Lab presidential vote data
- Applied shared `clean_acs()` function across all ACS tables
- Merged all tables on `county_fips` using inner joins
- Created binary target `party_winner` and continuous target `dem_vote_share`

### Bug Fix 1 — ACS Sentinel Value
The Census Bureau uses `-666666666` for suppressed county data. Replaced all occurrences with `NaN` before merging — without this fix all statistics on affected columns are incorrect.

### Bug Fix 2 — MIT Election Mode Filter
The reference implementation filtered only `mode == 'TOTAL'`, silently dropping 849 counties that exclusively report by individual voting modes (absentee, early voting etc.)

| Approach | Counties |
|----------|----------|
| TOTAL only (reference) | 2,305 |
| Fixed (TOTAL + mode fallback) | 3,154 |
| Final merged dataset | 3,115 |

### Final Dataset
- **Shape:** 3,115 counties × 162 features
- **Missing values:** 1 (median income, 1 county)
- **Republican-winning counties:** 2,576 (82.7%)
- **Democrat-winning counties:** 539 (17.3%)

---

## Part 2 — Exploratory Data Analysis

### Key Analyses
- Class imbalance: 4.78x (Republican vs Democrat counties)
- Swing county breakdown: 673 competitive counties (40–60% dem share)
- 8 engineered rate features from raw ACS counts
- Mutual information ranking — white_pct strongest predictor (0.112)
- Multicollinearity analysis (white_pct vs black_pct: -0.81)
- State-level aggregation — Oklahoma and West Virginia: 0% Democrat counties

### Feature Engineering

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

---

## Part 3 — Baseline Models (2020 Demographics Only)

### Classification Results

| Model | Accuracy | Dem F1 | ROC AUC |
|-------|----------|--------|---------|
| Logistic Regression | 0.883 | 0.727 | 0.952 |
| Random Forest | 0.929 | 0.788 | 0.960 |
| XGBoost | 0.917 | 0.768 | 0.961 |

### Regression Results

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Linear Regression | 0.669 | 0.094 | 0.074 |
| Ridge | 0.669 | 0.094 | 0.074 |
| Lasso | 0.671 | 0.094 | 0.074 |
| Random Forest | 0.737 | 0.084 | 0.063 |

### Clustering Results
- Algorithm: K-Means, optimal k=3 (silhouette: 0.309)
- Cluster 0 — Urban/Diverse (avg dem share: 0.50)
- Cluster 1 — Mixed/Swing (avg dem share: 0.42)
- Cluster 2 — Rural/White (avg dem share: 0.26)

---

## Part 4 — Historical Feature Engineering (Novel Contribution)

### Overview
Built a rich historical feature set using MIT Election Lab data from 2000–2020 to predict 2024 election outcomes. This goes significantly beyond demographics-only modeling.

### Historical Features (per election year 2000–2020)
- `dem_share_{year}` — Democratic vote share for each year
- `winner_{year}` — Binary winner for each year

### Engineered Political Trend Features

| Feature | Description |
|---------|-------------|
| `dem_wins_count` | Number of Democrat wins across 2000–2020 |
| `avg_dem_share` | Average dem vote share across all years |
| `dem_share_trend` | Overall direction: dem_share_2020 - dem_share_2000 |
| `momentum_2016_2020` | Recent shift: dem_share_2020 - dem_share_2016 |
| `momentum_2012_2016` | Prior shift: dem_share_2016 - dem_share_2012 |
| `vote_volatility` | Std dev of dem share across all years |
| `competitive_count` | Elections where county was within 5% of 50% |
| `political_alignment` | Strong Rep / Lean Rep / Swing / Lean Dem / Strong Dem |

### Interaction Features (unique to this work)

| Feature | Description |
|---------|-------------|
| `edu_x_wins` | higher_ed_rate × dem_wins_count |
| `edu_x_trend` | higher_ed_rate × dem_share_trend |
| `white_x_momentum` | white_pct × momentum_2016_2020 |
| `income_x_volatility` | median_income × vote_volatility |

**`edu_x_wins` ranked #4 in Random Forest feature importance** — more important than winner_2016, confirming that the combination of education and historical wins is a powerful predictor.

### Political Alignment Distribution
| Alignment | Counties |
|-----------|----------|
| Strong Republican | 2,078 |
| Lean Republican | 403 |
| Strong Democrat | 347 |
| Lean Democrat | 231 |
| Swing | 93 |

---

## Part 5 — Historical Classification (Predicting 2024)

**Target:** `party_winner_2024`  
**Training data:** 2000–2020 features  
**Test:** 2024 election results  
**Total features:** 32 (historical + demographic + interaction)

### Results

| Model | Accuracy | Dem F1 | Dem Precision | Dem Recall | ROC AUC |
|-------|----------|--------|---------------|------------|---------|
| Logistic Regression | 0.979 | 0.930 | 0.929 | 0.931 | 0.996 |
| Random Forest | 0.981 | 0.935 | 0.967 | 0.906 | 0.998 |
| Gradient Boosting | 0.984 | 0.944 | 0.966 | 0.922 | 0.998 |
| **XGBoost** | **0.986** | **0.952** | **0.989** | **0.917** | **0.998** |

### Ablation Study

| | Full Features | Demo Only |
|---|---|---|
| Accuracy | 0.986 | 0.934 |
| Dem F1 | 0.952 | 0.739 |
| ROC AUC | 0.998 | 0.960 |

---

## Part 6 — Historical Regression (Predicting 2024 Vote Share)

**Target:** `dem_share_2024` (continuous 0–1)

### Results

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Linear Regression | 0.9909 | 0.0149 | 0.0106 |
| Ridge | 0.9908 | 0.0149 | 0.0106 |
| Lasso | 0.9906 | 0.0151 | 0.0109 |
| Random Forest | 0.9905 | 0.0152 | 0.0109 |
| **Gradient Boosting** | **0.9912** | **0.0146** | **0.0103** |
| XGBoost | 0.9910 | 0.0148 | 0.0104 |

### Ablation Study

| | Full Features | Demo Only |
|---|---|---|
| R² | 0.9912 | 0.7344 |
| RMSE | 0.0146 | 0.0803 |
| Improvement | +0.2568 R² | 5.5x smaller RMSE |

---

## Part 7 — Historical Clustering

**Features:** 26 (historical + demographic + interaction)

### Results
- Optimal k=3
- Silhouette score: 0.183 (lower than demo-only due to more dimensions)
- **Political purity much higher than demographics-only clustering**

| Cluster | Profile | Avg Dem Share 2024 | Avg Wins | Purity |
|---------|---------|-------------------|----------|--------|
| 0 | Strong Democrat | 0.550 | 4.5/6 | 65.8% Dem |
| 1 | Strong Republican | 0.248 | 0.0/6 | 99.9% Rep |
| 2 | Republican (drifting) | 0.255 | 0.8/6 | 99.8% Rep |

Cluster 2 is the most interesting — avg dem_share_trend of -0.170 identifies counties that were once competitive but drifted heavily Republican over 20 years (classic "Obama → Trump" counties).

---

## Part 8 — Comprehensive Model Evaluation

### Full Comparison

| Task | Demo Only Best | Historical Best | Improvement |
|------|---------------|-----------------|-------------|
| Classification Accuracy | 0.929 (RF) | 0.986 (XGB) | +0.057 |
| Democrat F1 | 0.788 (RF) | 0.952 (XGB) | +0.164 |
| ROC AUC | 0.960 (RF) | 0.998 (XGB) | +0.038 |
| Regression R² | 0.737 (RF) | 0.991 (GB) | +0.254 |
| Regression RMSE | 0.084 (RF) | 0.015 (GB) | 5.5x smaller |

---

## Key Findings

1. **Historical voting behavior dominates prediction** — dem_share_2020 alone has 0.989 importance score in regression, explaining nearly everything about 2024 vote share
2. **Counties barely change** — past voting behavior almost perfectly predicts future behavior
3. **edu_x_wins interaction feature ranked #4** in classification importance — novel contribution not present in reference implementation
4. **Adding historical features improved F1 by +0.164 and R² by +0.254** vs demographics only
5. **Only 93 genuinely swing counties** exist across all 6 election cycles (2000–2020)
6. **Cluster 2 identifies "drifting Republican" counties** — avg trend of -0.170, classic Obama→Trump profile
7. **Demographics alone explain 73.7% of vote share** — strong signal even without history

---

## Challenges and Resolutions

| Challenge | Resolution |
|-----------|------------|
| ACS sentinel value -666666666 | Replaced all with NaN before merging |
| MIT mode filter dropping 849 counties | Fixed with TOTAL + mode fallback approach |
| NAME column misidentified as estimate column | Added explicit skip set in clean_acs() |
| 4.78x class imbalance | Used class_weight='balanced' for all classifiers |
| Raw counts biased by county size | Engineered normalized rate features |
| PerformanceWarning on fragmented DataFrame | Used df.copy() before feature engineering |

---

## Conclusions and Recommendations

- **Use XGBoost with historical features** for classification — 98.6% accuracy on 2024 prediction
- **Use Gradient Boosting with historical features** for regression — R² 0.991
- **dem_share_2020 is by far the most important single feature** — any model must include it
- **Interaction features add value** — edu_x_wins ranked #4 confirms non-linear relationships
- **Evaluate on the 93 swing counties separately** — these are the hardest to predict
- **Do not use white_pct and black_pct together** in linear models (correlation -0.81)
- **Demographics alone are a strong baseline** but historical features are essential for near-perfect prediction

---

*Siddhesh Dayanand Patil — DSC 445 Machine Learning 1 — DePaul University — May 2026*
