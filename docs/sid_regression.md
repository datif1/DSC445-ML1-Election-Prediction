# Sid — Regression Models

**Notebook:** `sid_regression_models.ipynb`  
**Branch:** `siddhesh`  
**Input:** `../data/processed/model_ready_dataset.csv`  
**Target:** `dem_vote_share` (continuous 0–1)

---

## Overview

Four regression models trained to predict the actual Democratic vote share percentage for each county. 80/20 train/test split with `random_state=42`.

---

## Features Used

```
higher_ed_rate, poverty_rate, unemployment_rate,
white_pct, black_pct, asian_pct,
median_income, log_population, housing_density
```

---

## Models

| Model | Key Parameters |
|-------|---------------|
| Linear Regression | OLS baseline |
| Ridge | alpha=1.0 |
| Lasso | alpha=0.001 |
| Random Forest Regressor | 200 trees |

---

## Test Set Results

| Model | R² | RMSE | MAE | CV R² Mean |
|-------|-----|------|-----|------------|
| Linear Regression | 0.6688 | 0.0943 | 0.0739 | 0.673 |
| Ridge | 0.6688 | 0.0943 | 0.0738 | 0.673 |
| Lasso | 0.6708 | 0.0940 | 0.0736 | 0.672 |
| Random Forest | 0.7367 | 0.0841 | 0.0629 | 0.743 |

---

## Feature Importance — Random Forest

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | white_pct | 0.2701 |
| 2 | higher_ed_rate | 0.2255 |
| 3 | asian_pct | 0.1990 |
| 4 | log_population | 0.1247 |
| 5 | black_pct | 0.0448 |
| 6 | housing_density | 0.0398 |
| 7 | poverty_rate | 0.0376 |
| 8 | unemployment_rate | 0.0303 |
| 9 | median_income | 0.0283 |

---

## Key Findings

- **Best model: Random Forest** — R² 0.737, RMSE 0.084, MAE 0.063
- Linear, Ridge, Lasso all identical at R² 0.669 — regularization provides no benefit
- Random Forest gains +0.068 R² by capturing non-linear demographic interactions
- RMSE of 0.084 means predictions are on average ~8.4% off
- white_pct is top feature for regression (0.270) vs higher_ed_rate for classification (0.229)
- median_income is the weakest predictor (0.028) despite appearing important in raw correlations
