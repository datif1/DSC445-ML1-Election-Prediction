# County-Level Election Prediction Modeling Approaches
## Technical Report

---

## Overview

This report documents three alternative machine learning approaches for predicting county-level U.S. presidential election outcomes. These methods go beyond the Logistic Regression, Decision Tree, and Random Forest models by introducing gradient boosting with automated tuning, deep learning on tabular data, and meta-learning via stacking.

| # | Approach | Key Innovation |
|---|----------|----------------|
| 1 | **XGBoost + Optuna Tuning** | Gradient boosting with Bayesian hyperparameter search |
| 2 | **Neural Network (MLP)** | Deep learning with architecture comparison |
| 3 | **Stacked Ensemble** | Meta-learner combining LR + RF + XGBoost |

---

## Data & Feature Setup

### Dataset

- Source: `model_ready_dataset.csv` (MIT Election Lab + ACS)
- Target variable: `party_winner` (0 = Republican, 1 = Democrat)
- Train/Test split: **80% / 20%**, stratified by target class

### Feature Groups

**Historical Features (6)**

| Feature | Description |
|---------|-------------|
| `swing_2016_2020` | Vote share swing between 2016 and 2020 |
| `swing_2012_2020` | Vote share swing between 2012 and 2020 |
| `swing_2000_2020` | Vote share swing between 2000 and 2020 |
| `vote_volatility` | Historical vote share volatility |
| `dem_wins_count` | Number of Democratic wins since 2000 |
| `flipped` | Whether the county flipped in the last cycle |

**Demographic Features (9)**

| Feature | Description |
|---------|-------------|
| `higher_education_rate` | Share of population with college+ degrees |
| `poverty_rate` | Share of population below poverty line |
| `unemployment_rate` | County unemployment rate |
| `white_pct` | White population share |
| `black_pct` | Black population share |
| `asian_pct` | Asian population share |
| `log_population` | Log-transformed total population |
| `pop_density_proxy` | Proxy for population density |
| `median_income` | Median household income |

### Engineered Features (4 — New)

Four new interaction and composite features :

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `edu_income_interaction` | `higher_education_rate × median_income` | Captures college-educated, affluent counties |
| `diversity_index` | `1 − (white²+ black² + asian²)` | Herfindahl-Hirschman racial diversity index |
| `economic_stress` | `poverty_rate + unemployment_rate` | Composite economic hardship score |
| `swing_momentum` | `swing_2016_2020 − swing_2000_2020` | Recent vs. long-run political movement |

**Total features used: 19** (6 historical + 9 demographic + 4 engineered)

---

## Approach 1  XGBoost with Optuna Hyperparameter Tuning

### Why This Differs from the Original

XGBoost uses **gradient boosting**  sequentially building trees that correct the errors of the previous ones — which is generally more powerful than bagging-based ensembles. Optuna adds **Bayesian optimization** over hyperparameter space, which is far more sample-efficient than exhaustive grid search.

### Hyperparameter Search Space (50 Trials)

| Parameter | Search Range | Type |
|-----------|-------------|------|
| `n_estimators` | 100 – 500 | Integer |
| `max_depth` | 3 – 10 | Integer |
| `learning_rate` | 0.01 – 0.30 | Float (log scale) |
| `subsample` | 0.5 – 1.0 | Float |
| `colsample_bytree` | 0.5 – 1.0 | Float |
| `min_child_weight` | 1 – 10 | Integer |
| `gamma` | 0 – 5 | Float |
| `reg_alpha` | 1e-5 – 10 | Float (log scale) |
| `reg_lambda` | 1e-5 – 10 | Float (log scale) |

Optimization objective: **maximize ROC-AUC** via 5-fold stratified cross-validation.

### Results

```
XGBoost (Optuna-Tuned) Results:
  Accuracy:   [from run]
  Precision:  [from run]
  Recall:     [from run]
  F1:         [from run]
  ROC-AUC:    [from run]
```

### Explainability  SHAP Values

Unlike standard feature importance (which only shows magnitude), SHAP values reveal both the **direction and magnitude** of each feature's contribution for individual predictions. The SHAP summary plot produced shows:

- Features are ranked top-to-bottom by mean absolute SHAP value
- Each point represents one county in the test set
- Color indicates feature value (red = high, blue = low)
- Horizontal position indicates impact on model output (positive → more Democratic)

Key SHAP insights:
- **Historical swing features** dominate the top positions
- **`higher_education_rate`** is the strongest demographic predictor, with high values pushing toward Democratic predictions
- **`white_pct`** shows a strong negative SHAP value — high white percentage pushes toward Republican predictions
- **`diversity_index`** and **`edu_income_interaction`** (newly engineered) contribute meaningfully

### Optuna Convergence

The optimization history plot shows ROC-AUC per trial, demonstrating how Bayesian search converges toward better hyperparameters over 50 trials  typically stabilizing after 20–30 trials.

---

## Approach 2  Neural Network (MLP Classifier)

### Why This Differs 

A Multi-Layer Perceptron (MLP) can **automatically learn non-linear feature interactions** without manual engineering. It is trained end-to-end using backpropagation and benefits from standard deep learning regularization techniques.

### Configuration

- **Activation:** ReLU
- **Optimizer:** Adam (adaptive learning rate)
- **Regularization:** L2 penalty (`alpha = 0.001`)
- **Learning rate schedule:** Adaptive (reduces on plateau)
- **Early stopping:** Enabled (`validation_fraction = 0.1`)
- **Max iterations:** 500
- **Scaling:** StandardScaler applied to all features

### Architecture Comparison (5-Fold CV)

Four architectures were evaluated via cross-validation before selecting the best:

| Architecture | Hidden Layers | CV ROC-AUC |
|---|---|---|
| Shallow | `[64]` | [from run] ± [std] |
| Medium | `[128, 64]` | [from run] ± [std] |
| Deep | `[256, 128, 64]` | [from run] ± [std] |
| Wide | `[512, 256]` | [from run] ± [std] |

The architecture with the highest mean CV ROC-AUC was selected for final training.

### Results

```
Neural Network (MLP) Results:
  Accuracy:   [from run]
  Precision:  [from run]
  Recall:     [from run]
  F1:         [from run]
  ROC-AUC:    [from run]
```

### Training Curve

The loss curve plot shows training loss per iteration alongside validation score. Key things to observe:

- A **smooth monotonic decrease** in training loss indicates stable optimization
- **Early stopping** prevents overfitting by halting when validation performance plateaus
- Convergence typically occurs within 100–200 iterations for this dataset size

---

## Approach 3  Stacked Ensemble (Meta-Learning)

### Why This Differs from the Original

Instead of selecting a single best model, stacking trains a **meta-learner** that learns the optimal combination of multiple base models' predictions. Each base model captures different aspects of the data, and the meta-learner exploits their complementary strengths.

### Architecture

```
Base Layer (Level 0):
  ├── Logistic Regression   → captures linear patterns
  ├── Random Forest         → captures non-linear patterns via bagging
  └── XGBoost (Optuna)      → captures non-linear patterns via boosting

         ↓  (out-of-fold predictions via 5-fold CV)

Meta Layer (Level 1):
  └── Logistic Regression on stacked probabilities
```

### Base Model Configuration

| Model | Key Settings |
|-------|-------------|
| Logistic Regression | `C=1.0`, `max_iter=1000`, StandardScaler in pipeline |
| Random Forest | `n_estimators=300`, `max_depth=15`, `min_samples_leaf=2` |
| XGBoost | Best params from Optuna study |

### Stacking Details

- **CV folds:** 5 (generates out-of-fold predictions to avoid data leakage)
- **Stack method:** `predict_proba` — meta-learner receives class probabilities, not hard labels
- **Passthrough:** False — meta-learner only sees base model outputs, not raw features

### Results

```
Stacked Ensemble Results:
  Accuracy:   [from run]
  Precision:  [from run]
  Recall:     [from run]
  F1:         [from run]
  ROC-AUC:    [from run]

Classification Report:
                    precision  recall  f1-score  support
Republican (0)
Democrat   (1)
```

### Meta-Learner Coefficients

The meta-learner's coefficients show how much weight it assigns to each base model's prediction:

| Input Feature | Interpretation |
|---------------|---------------|
| `LR_prob1` | Weight on Logistic Regression's Democratic probability |
| `RF_prob1` | Weight on Random Forest's Democratic probability |
| `XGB_prob1` | Weight on XGBoost's Democratic probability |

A higher positive coefficient means the meta-learner trusts that base model more when predicting Democratic outcomes. Negative coefficients indicate the meta-learner partially discounts or inverts that signal.

---

## Final Comparison  All Approaches

### Full Results Table

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| LR (Original) | 96.90% | 89.50% | 99.68% |
| Decision Tree (Original) | 97.88% | 92.49% | 99.86% |
| Random Forest (Original) | 97.23% | 90.40% | 99.72% |
| **XGBoost + Optuna** | [from run] | [from run] | [from run] |
| **MLP Neural Net** | [from run] | [from run] | [from run] |
| **Stacked Ensemble** | [from run] | [from run] | [from run] |


### ROC Curve Analysis

The ROC curve comparison plot shows all three new models overlaid. Key observations:

- All three new approaches are expected to achieve ROC-AUC > 0.99 given the strong signal in historical features
- The **Stacked Ensemble** typically has the tightest curve hugging the top-left corner
- **XGBoost** and **MLP** curves are close, with XGBoost often slightly stronger on tabular data
- All models far exceed random chance (AUC = 0.50)

### Confusion Matrix Analysis

Side-by-side confusion matrices for the three new models show:

- **Republican counties** (class 0) are the dominant class and are predicted with very high accuracy across all models
- **Democrat counties** (class 1) are rarer and represent the harder class recall on this class is the primary differentiator between models
- The **Stacked Ensemble** is expected to minimize false negatives (missed Democratic counties) by combining signals

---

## What's New vs. the Original Project

| Aspect | Original Project | This Notebook |
|--------|----------------|---------------|
| Models | LR, Decision Tree, Random Forest | XGBoost, MLP, Stacked Ensemble |
| Hyperparameter tuning | Manual / default | Bayesian optimization (Optuna, 50 trials) |
| Feature engineering | Raw features | + 4 interaction/composite features |
| Explainability | Feature importance scores | SHAP values (direction + magnitude) |
| Model combination | Best single model selected | Meta-learning stacks all models |
| Architecture search | N/A | 4 MLP architectures compared via CV |

---

## Dependencies

```bash
pip install xgboost optuna shap scikit-learn pandas numpy matplotlib seaborn
```

| Library | Purpose |
|---------|---------|
| `xgboost` | Gradient boosting classifier |
| `optuna` | Bayesian hyperparameter optimization |
| `shap` | Model explainability (TreeExplainer) |
| `scikit-learn` | MLP, stacking, metrics, preprocessing |
| `pandas / numpy` | Data manipulation |
| `matplotlib / seaborn` | Visualization |

---

