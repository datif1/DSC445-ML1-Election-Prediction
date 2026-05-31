# Sid — Classification Models

**Notebook:** `sid_classification.ipynb`  
**Branch:** `siddhesh`  
**Input:** `../data/processed/model_ready_dataset.csv`  
**Target:** `party_winner` (0 = Republican, 1 = Democrat)

---

## Overview

Three classification models trained to predict the binary county winner. All models use `class_weight='balanced'` to handle the 4.78x class imbalance. Features standardized with `StandardScaler`. 80/20 stratified train/test split with `random_state=42`.

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
| Logistic Regression | class_weight='balanced', max_iter=1000 |
| Random Forest | 200 trees, class_weight='balanced' |
| XGBoost | 200 estimators, scale_pos_weight=4.78 |

---

## Test Set Results

| Model | Accuracy | Dem F1 | Dem Precision | Dem Recall | ROC AUC |
|-------|----------|--------|---------------|------------|---------|
| Logistic Regression | 0.883 | 0.727 | 0.610 | 0.898 | 0.952 |
| Random Forest | 0.929 | 0.788 | 0.820 | 0.759 | 0.960 |
| XGBoost | 0.917 | 0.768 | 0.741 | 0.796 | 0.961 |

---

## 5-Fold Cross-Validation (F1 — Democrat class)

| Model | CV F1 Mean | CV F1 Std |
|-------|------------|-----------|
| Logistic Regression | 0.687 | 0.022 |
| Random Forest | 0.749 | 0.037 |
| XGBoost | 0.758 | 0.037 |

---

## Feature Importance — Random Forest

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | higher_ed_rate | 0.2295 |
| 2 | white_pct | 0.1979 |
| 3 | log_population | 0.1270 |
| 4 | asian_pct | 0.1155 |
| 5 | black_pct | 0.1019 |
| 6 | poverty_rate | 0.0654 |
| 7 | median_income | 0.0582 |
| 8 | unemployment_rate | 0.0552 |
| 9 | housing_density | 0.0493 |

---

## Key Findings

- **Best model: Random Forest** — highest accuracy (0.929), highest Dem F1 (0.788), highest Precision (0.820)
- XGBoost edges out on CV F1 (0.758) suggesting slightly better generalization
- All models achieve AUC > 0.95 — excellent discrimination ability
- Feature importance confirms EDA mutual information ranking
