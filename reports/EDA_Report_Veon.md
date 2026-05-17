# Exploratory Data Analysis Report

**Predicting U.S. County-Level Presidential Voting Outcomes**  
Using Demographic and Socioeconomic Machine Learning Models

**Author: Veon Ivon Almeida**  
Project Team: Shang Andrews, Mohammed Atif Diwan, Yung Han Jeong, Siddhesh Dayanand Patil  
DSC 445 — Machine Learning I | May 2025

---

| **3,115 Counties** Final merged | **166 Features** After merging | **7 Election Years** 2000–2024 | **4 Missing Values** After cleaning | **84 Flipped Counties** 2016→2020 |
|---|---|---|---|---|

---

## 1. Dataset Overview

This EDA covers three merged county-level datasets: the MIT Election Lab Presidential Returns (2000–2024), and the ACS 5-Year Demographic and Socioeconomic Estimates (2020). The final merged dataset contains 3,115 U.S. counties with 166 columns.

### 1.1 Raw Dataset Shapes

| Dataset | Shape | Notes |
|---|---|---|
| countypres_2000-2024.csv | (94,151 × 12) | All election years, all parties |
| ACSDT5Y2020 B01001 — Age | (3,222 × 101) | Before cleaning |
| ACSDT5Y2020 B02001 — Race | (3,222 × 23) | Before cleaning |
| ACSDT5Y2020 B15003 — Education | (3,222 × 53) | Before cleaning |
| ACSDT5Y2020 B19013 — Income | (3,222 × 5) | Before cleaning |
| ACSDT5Y2020 B17001 — Poverty | (3,222 × 121) | Before cleaning |
| ACSDT5Y2020 B23025 — Employment | (3,222 × 17) | Before cleaning |
| ACSDT5Y2020 B25001 — Housing | (3,222 × 5) | Before cleaning |
| Final Merged Dataset | (3,115 × 166) | ACS + Election 2020 + Swing features |

### 1.2 Data Types in Final Dataset

The final merged dataset contains the following column types:

- **int64:** 156 columns (raw ACS estimate counts)
- **float64:** 9 columns (vote shares, swing features, engineered rates)
- **object:** 1 column (NAME — all null, to be dropped before modeling)

### 1.3 Missing Values

The dataset is nearly complete. Only 4 columns contain any missing values:

| Column | Missing | Resolution |
|---|---|---|
| NAME | 3,115 (100%) | Drop — fully empty column |
| swing_2000_2020 | 2 | Drop rows or impute with 0 |
| swing_2012_2020 | 1 | Drop rows or impute with 0 |
| B19013_001E (Median Income) | 1 | Imputed with state-level median |

> The income sentinel value of -666,666,666 (ACS suppressed data code) was detected and replaced with NaN, then imputed using the state-level median. This is the only substantive imputation performed.

---

## 2. Election Data Analysis

### 2.1 Target Variable — 2020 County Winner

The primary classification target is `party_winner` (1 = Democrat, 0 = Republican), derived from 2020 county-level vote counts. The dataset has a significant class imbalance:

| Party | Counties | Share | Implication |
|---|---|---|---|
| Republican (0) | **2,576** | 82.7% | Majority class — risk of bias |
| Democrat (1) | **539** | 17.3% | Minority class — needs handling |

> Imbalance handling recommendation: Use `class_weight='balanced'` in sklearn models and evaluate with F1-score and balanced accuracy rather than raw accuracy. A naive model predicting Republican for every county would achieve 82.7% accuracy while being completely useless.

### 2.2 Democratic Vote Share Distribution

The regression target `dem_vote_share` (continuous, 0–1) shows a right-skewed distribution:

- **Mean: 0.339** — the average county gives Democrats about 34% of the two-party vote
- Median is lower than mean, confirming the right skew
- Most counties cluster between 0.1 and 0.45, well below the 0.5 threshold
- Only a small tail of counties exceeds 0.5, reflecting the Democrat minority

This distribution explains why classification is hard — the data is not centered around 0.5 and there is no natural decision boundary from the data alone.

### 2.3 Historical Vote Trends (2000–2024)

Because all 7 election years are included, trend-based features were engineered from the full history. The per-year average county Democratic share shows a consistent pattern:

| 2000 | 2004 | 2008 | 2012 | 2016 | 2020 | 2024 |
|---|---|---|---|---|---|---|
| ~0.41 | ~0.38 | ~0.40 | ~0.39 | ~0.34 | ~0.34 | ~0.33 |

The long-term trend shows counties trending more Republican since 2000, with a notable drop between 2012 and 2016. The 2020 election showed minimal overall change from 2016 (average swing of +0.55 toward Democrats).

### 2.4 Swing and Volatility

| Feature | Value | Interpretation |
|---|---|---|
| Avg swing 2016 → 2020 | +0.0055 | Slight Dem improvement, mostly stable |
| Counties that flipped | 84 of 3,115 | Only 2.7% changed party |
| R → D flips | 69 counties | More common direction in 2020 |
| D → R flips | 15 counties | Rare — Dems held most their counties |
| Vote volatility (avg std dev) | ~0.05 | Very low — counties vote consistently |
| Dem wins count = 0 (never Dem) | ~2,200+ | Structural Republican counties |
| Dem wins count = 7 (always Dem) | ~200+ | Structural Democratic counties |

> **The extremely low flip rate (2.7%) confirms that historical voting behavior is the strongest predictor — counties that have voted Republican for 7 consecutive elections are very unlikely to flip, regardless of demographics.**

---

## 3. Demographic and Socioeconomic Features

### 3.1 Engineered Feature Distributions

Raw ACS counts were converted to normalized rates to allow meaningful comparison across counties of different sizes. The key engineered features and their summary statistics are:

| Feature | Mean | Std | Min | 25% | Median | Max |
|---|---|---|---|---|---|---|
| Higher education rate | 0.16 | 0.07 | 0.00 | 0.11 | 0.14 | 0.56 |
| Poverty rate | 0.14 | 0.06 | 0.00 | 0.10 | 0.13 | 0.58 |
| Unemployment rate | 0.05 | 0.03 | 0.00 | 0.04 | 0.05 | 0.30 |
| White population % | 0.82 | 0.17 | 0.05 | 0.75 | 0.88 | 1.00 |
| Black population % | 0.09 | 0.14 | 0.00 | 0.01 | 0.02 | 0.88 |
| Asian population % | 0.01 | 0.03 | 0.00 | 0.00 | 0.01 | 0.43 |
| Log population | 10.29 | 1.49 | 4.77 | 9.32 | 10.17 | 16.12 |
| Pop density proxy | 2.11 | 0.39 | 0.45 | 1.90 | 2.14 | 3.91 |

### 3.2 Key Demographic Findings

**Race Composition**

- The average county is 82% white — but this masks enormous variation (std: 17%)
- Some counties are 100% white, others are as low as 5% white
- Black population averages only 9% but ranges from 0% to 88% — highly concentrated in Southern counties
- Asian population is small on average (1.4%) but reaches 43% in some urban California/Hawaii counties

**Education**

- Mean higher education rate is only 16% — most U.S. counties have a low college-educated share
- The distribution is right-skewed — a few urban/suburban counties (max 56%) pull the mean up
- The inter-quartile range is 11%–18%, meaning most counties cluster in a narrow band

**Economic Indicators**

- Average poverty rate is 14% — substantial variation from 0% to 58%
- Unemployment averages 5% — relatively low spread (std: 3%), with outliers up to 30%
- Median household income was imputed for 1 county (ACS suppressed value)

**Population**

- Log-transformed due to extreme right skew — Los Angeles County has 10 million residents while the smallest county has 117
- Population density proxy (persons per housing unit) averages 2.1 — very stable across counties (std: 0.39)

---

## 4. Feature Correlations with Target

### 4.1 Top Positive Correlations with Democratic Vote Share

| Feature | r | Dir | Strength |
|---|---|---|---|
| dem_wins_count (historical Dem wins) | 0.82 | + | ██████ |
| log_population (county size) | 0.54 | + | ████ |
| higher_education_rate | 0.51 | + | ████ |
| black_pct | 0.44 | + | ███ |
| asian_pct | 0.44 | + | ███ |
| swing_2012_2020 | 0.39 | + | ███ |
| democrat_votes (raw count) | 0.39 | + | ███ |

### 4.2 Top Negative Correlations with Democratic Vote Share

| Feature | r | Dir | Strength |
|---|---|---|---|
| white_pct | -0.57 | - | ████ |
| vote_volatility | -0.30 | - | ██ |
| poverty_rate | -0.08 | - | — |

### 4.3 Interpretation

Several important findings emerge from the correlation analysis:

- **`dem_wins_count` is the strongest single predictor (r = 0.82), confirming that historical voting behavior dominates. This is the key novel feature enabled by using all 7 years of election data.**
- Urban counties vote more Democratic (`log_population` r = 0.54). County population size is a strong structural predictor.
- Education is a strong positive predictor (r = 0.51). The college education divide has become a defining feature of American elections.
- Race composition is highly predictive — `white_pct` is the strongest negative predictor (r = -0.57), while `black_pct` and `asian_pct` are positive.
- Poverty rate has a surprisingly weak correlation (r = -0.08). Poor counties do not uniformly vote Democratic, reflecting the rural/urban divide.
- Vote volatility is negatively correlated (r = -0.30). Highly volatile counties tend to lean Republican — Democrats hold consistent urban strongholds.
- `swing_2000_2020` (r = 0.66) captures long-run realignment — counties that have moved toward Democrats over 20 years are likely current or future Dem counties.

### 4.4 Multicollinearity Concerns

Several highly correlated feature pairs may cause issues in linear models:

- `white_pct` and `black_pct` are strongly negatively correlated (by construction) — use only one or compute net non-white percentage
- `dem_wins_count` and `dem_vote_share` are highly correlated (r = 0.82) — `dem_wins_count` should be excluded from regression targets but kept for classification
- Raw ACS count variables (e.g., `B15003_025E`) are nearly perfectly correlated with their engineered counterparts — use only the engineered rate versions
- `democrat_votes` correlates with `log_population` and `dem_vote_share` — remove from feature set to avoid data leakage

---

## 5. Implications for Modeling

### 5.1 Class Imbalance

The 83/17 class split is the most important data quality issue. Recommendations:

- Use `class_weight='balanced'` in all sklearn classifiers (LogReg, RF, SVM, XGBoost)
- Evaluate with F1-score (weighted or macro), balanced accuracy, and ROC-AUC — not raw accuracy
- Consider SMOTE oversampling as an alternative — synthesizes new minority-class examples rather than reweighting
- Report confusion matrix separately for each class — distinguish between false positives and false negatives on Democrat counties

### 5.2 Feature Selection Recommendations

Based on EDA findings, the recommended feature set for modeling is:

- **Historical:** `dem_wins_count`, `swing_2016_2020`, `swing_2012_2020`, `swing_2000_2020`, `vote_volatility`, `dem_share_2016`
- **Demographic:** `higher_education_rate`, `white_pct`, `black_pct`, `asian_pct`, `log_population`, `pop_density_proxy`
- **Socioeconomic:** `median_income`, `poverty_rate`, `unemployment_rate`
- **Drop:** `NAME`, raw ACS count columns, `democrat_votes`, `republican_votes` (data leakage)

### 5.3 Regression vs Classification

Both targets should be modeled:

- **Regression on `dem_vote_share`:** the continuous target avoids class imbalance issues and produces richer predictions — use MSE as primary metric
- **Classification on `party_winner`:** the binary target is needed for Electoral College approximation — use F1 and balanced accuracy
- The regression output can inform classification by thresholding: if predicted `dem_vote_share > 0.5`, classify as Democrat

### 5.4 Data Leakage Checks

The following columns must be excluded from all model training to prevent leakage:

- `democrat_votes`, `republican_votes`, `totalvotes` — these directly encode the target
- `party_winner` when training regression models on `dem_vote_share`
- `dem_vote_share` when training classification models on `party_winner`
- `dem_share_2020` — same year as the target; only prior years should be used as predictors

### 5.5 Suggested Next Steps

- Scale features using `StandardScaler` or `MinMaxScaler` before fitting linear models
- Run baseline models (Logistic Regression, Decision Tree) before moving to Random Forest and XGBoost
- Use K-Means clustering on the demographic features to discover county archetypes, then overlay voting outcomes
- Perform SHAP analysis on tree-based models to confirm which features drive individual predictions

---

*End of EDA Report — Veon Ivon Almeida — DSC 445*
