# Regression Modeling Summary

## Objective

The objective of the regression analysis was to predict the Democratic vote share in the 2024 U.S. Presidential Election at the county level. Unlike the classification models, which predicted the winning party, regression models estimate the exact Democratic vote share for each county.

The target variable used throughout this section was:

- target_2024_dem_vote_share

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Lower MAE and RMSE values indicate better prediction accuracy, while higher R² values indicate better explanatory power.

---

# Linear Regression

Linear Regression was used as the baseline regression model. This model assumes a linear relationship between the predictor variables and the target vote share.

### Results

| Metric | Value |
|----------|----------|
| MAE | 0.0723 |
| RMSE | 0.1015 |
| R² | 0.7008 |

### Findings

The model explained approximately 70.1% of the variation in county-level Democratic vote share. While the model produced reasonable predictions, residual plots indicated that the relationship between the predictors and target variable is not entirely linear.

---

# Ridge Regression

Ridge Regression extends Linear Regression by introducing L2 regularization, which reduces coefficient magnitude and helps address multicollinearity among features.

### Results

| Metric | Value |
|----------|----------|
| MAE | 0.0703 |
| RMSE | 0.0987 |
| R² | 0.7171 |

### Findings

Ridge Regression slightly improved prediction accuracy over the baseline Linear Regression model. The regularization penalty helped stabilize coefficients and reduced overfitting.

---

# Lasso Regression

Lasso Regression uses L1 regularization, which can shrink some coefficients to exactly zero and perform automatic feature selection.

### Results

| Metric | Value |
|----------|----------|
| MAE | 0.0698 |
| RMSE | 0.0987 |
| R² | 0.7172 |

### Findings

Lasso Regression achieved nearly identical performance to Ridge Regression while reducing the number of active predictors from 190 features to only 28 selected features. This indicates that many variables contributed little predictive information.

---

# Random Forest Regressor

Random Forest Regressor uses an ensemble of decision trees and averages their predictions to improve accuracy and reduce variance.

### Results

| Metric | Value |
|----------|----------|
| MAE | 0.0638 |
| RMSE | 0.0930 |
| R² | 0.7491 |

### Findings

Random Forest significantly outperformed all linear models. Historical voting variables emerged as the most important predictors, particularly previous Democratic vote shares and long-term voting trends.

The model captured nonlinear relationships that linear models were unable to represent.

---

# XGBoost Regressor

XGBoost Regressor is a gradient boosting algorithm that sequentially builds trees to correct previous prediction errors. It is one of the most powerful machine learning algorithms for structured tabular data.

### Results

| Metric | Value |
|----------|----------|
| MAE | 0.0590 |
| RMSE | 0.0879 |
| R² | 0.7760 |

### Findings

XGBoost achieved the best overall performance among all regression models.

The most influential variables included:

- dem_share_2016
- dem_share_2020
- dem_wins_count_2000_2020
- avg_dem_share_2000_2020

These results demonstrate that historical voting behavior is the strongest predictor of future county-level election outcomes.

---

# Regression Model Comparison

| Model | MAE | RMSE | R² |
|----------|----------|----------|----------|
| XGBoost Regressor | 0.0590 | 0.0879 | 0.7760 |
| Random Forest Regressor | 0.0638 | 0.0930 | 0.7491 |
| Lasso Regression | 0.0698 | 0.0987 | 0.7172 |
| Ridge Regression | 0.0703 | 0.0987 | 0.7171 |
| Linear Regression | 0.0723 | 0.1015 | 0.7008 |

---

# Overall Conclusion

The regression experiments demonstrated that tree-based ensemble methods substantially outperform traditional linear models when predicting county-level Democratic vote share.

Among all models evaluated, XGBoost Regressor achieved the highest predictive performance, obtaining the lowest prediction errors and the highest R² value. Historical voting variables consistently appeared as the most influential predictors, indicating that long-term voting patterns are highly informative for forecasting future election outcomes.

These findings align closely with the results obtained in the classification experiments, where historical election variables also dominated feature importance rankings.