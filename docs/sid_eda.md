# Sid — Exploratory Data Analysis

**Notebook:** `sid_eda.ipynb`  
**Branch:** `siddhesh`  
**Input:** `../data/processed/merged_county_dataset.csv`  
**Output:** `../data/processed/model_ready_dataset.csv`

---

## Overview

EDA on the merged county-level dataset. Covers class balance, swing county analysis, feature engineering, correlation analysis, mutual information ranking, and state-level patterns.

---

## Key Findings

### Class Imbalance

| Class | Counties | Percentage |
|-------|----------|------------|
| Republican (0) | 2,576 | 82.7% |
| Democrat (1) | 539 | 17.3% |
| Imbalance ratio | 4.78x | — |

Accuracy alone is not a reliable metric. Must use F1-score, Precision, Recall, and Confusion Matrix.

---

### Swing County Analysis

| Category | Vote Share | Counties | Share |
|----------|------------|----------|-------|
| Solid Republican | < 40% | 2,204 | 70.8% |
| Swing | 40%–60% | 673 | 21.6% |
| Solid Democrat | > 60% | 238 | 7.6% |

---

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

### Mutual Information Ranking

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

---

### Correlation Highlights

- `white_pct` vs `black_pct`: **-0.81** — multicollinearity warning
- `poverty_rate` vs `median_income`: **-0.72** — multicollinearity warning
- `white_pct` vs `dem_vote_share`: **-0.58** — strongest linear predictor
- `log_population` vs `dem_vote_share`: **+0.54**

---

### State-Level Patterns

Most Democrat-leaning states by avg county dem vote share:
- DC (0.921), Massachusetts (0.664), Hawaii (0.648)

Most Republican-leaning states:
- Nebraska (0.199), Oklahoma (0.203), West Virginia (0.243)
- Oklahoma and West Virginia had **0% Democrat-winning counties**

---

### EDA Summary Stats

| Metric | Value |
|--------|-------|
| Total counties | 3,115 |
| Swing counties (40-60%) | 673 |
| Avg dem vote share | 0.333 |
| Avg median income | $54,917 |
| Avg higher ed rate | 15.6% |
| Avg poverty rate | 14.1% |
| Top MI feature | white_pct (0.112) |
| Final dataset shape | 3,115 × 184 |
