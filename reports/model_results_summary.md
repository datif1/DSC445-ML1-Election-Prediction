# Model Results Summary

## Project Overview

This project aimed to predict county-level United States presidential election outcomes using demographic, socioeconomic, and historical political data. The primary objective was to investigate how strongly historical voting behavior and county demographic characteristics influence future election outcomes. Multiple machine learning classification models were trained and evaluated to predict the 2024 county-level presidential election results.

The dataset combined:
- MIT Election Lab county-level election results (2000–2024)
- American Community Survey (ACS) demographic and socioeconomic data

The project focused on:
- historical political trends,
- demographic feature engineering,
- model comparison,
- feature importance analysis,
- and ablation experiments.

---

# Models Implemented

The following classification models were implemented:

## Full Historical Models

These models used:
- historical election outcomes,
- historical Democratic vote shares,
- swing features,
- and demographic/socioeconomic variables.

Models:
- Logistic Regression
- Decision Tree
- Random Forest

---

## Demographics-Only Models

These models excluded all historical political features and used only:
- education,
- race composition,
- poverty,
- unemployment,
- income,
- and ACS demographic variables.

Models:
- Logistic Regression
- Decision Tree
- Random Forest

---

# Overall Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 96.90% | 81.00% | 100.00% | 89.50% | 99.68% |
| Decision Tree | 97.88% | 86.96% | 98.77% | 92.49% | 99.86% |
| Random Forest | 97.23% | 83.33% | 98.77% | 90.40% | 99.72% |
| Demo-Only Logistic Regression | 87.77% | 52.27% | 85.19% | 64.79% | 94.13% |
| Demo-Only Decision Tree | 87.77% | 52.24% | 86.42% | 65.12% | 87.84% |
| Demo-Only Random Forest | 93.64% | 75.61% | 76.54% | 76.07% | 96.36% |

---

# Key Findings

## 1. Historical Political Behavior Was the Strongest Predictor

The models that used historical election information consistently achieved the highest performance. Features such as:
- previous county winners,
- Democratic vote shares,
- swing trends,
- and historical averages

dominated the prediction process.

This demonstrates that county-level political behavior in the United States shows strong long-term continuity across election cycles.

---

## 2. Decision Tree Achieved the Best Overall Performance

Among all full-feature models, the Decision Tree classifier produced the strongest overall results:
- Accuracy: 97.88%
- F1-score: 92.49%
- ROC-AUC: 99.86%

The model successfully captured voting patterns with very few classification errors and demonstrated excellent predictive capability at the county level.

---

## 3. Demographic Features Still Retained Strong Predictive Power

Even after removing all historical political features, the models continued to achieve meaningful predictive performance.

Most notably:
- the demographics-only Random Forest achieved 93.64% accuracy,
- with a ROC-AUC of 96.36%.

This demonstrates that demographic and socioeconomic county characteristics alone still contain substantial information about political alignment.

---

## 4. Random Forest Performed Best for Demographics-Only Prediction

The Random Forest model significantly outperformed the other demographics-only models.

This suggests that:
- demographic relationships are highly nonlinear,
- and ensemble tree-based models are more effective at capturing complex interactions between variables such as education, race, poverty, and income.

---

# Feature Importance Analysis

## Historical Feature Importance

The most important historical features included:
- winner_2020
- winner_2016
- dem_share_2020
- dem_share_2016
- dem_wins_count_2000_2020

These variables overwhelmingly dominated prediction performance in the full historical models.

---

## Demographic Feature Importance

The most influential demographic features included:
- higher_education_rate
- white_pct
- black_pct
- asian_pct
- poverty_rate
- median_income
- doctorate_degree
- bachelors_degree
- masters_degree

These findings strongly align with modern political science research on county-level voting behavior in the United States.

---

# Major Political Insights

## Education Level Was the Strongest Demographic Predictor

Higher education-related variables consistently emerged as the most influential demographic predictors of Democratic voting behavior.

Counties with:
- higher rates of bachelor’s,
- master’s,
- and doctoral degrees

showed stronger tendencies toward Democratic alignment.

---

## Race Composition Strongly Influenced Political Alignment

Race-related variables such as:
- white_pct,
- black_pct,
- and asian_pct

were highly important predictors in the demographics-only models.

This indicates that county racial composition significantly contributes to political outcomes.

---

## Socioeconomic Variables Also Contributed Meaningfully

Variables such as:
- poverty rate,
- median income,
- and unemployment

showed measurable predictive influence, although weaker than historical political features.

---

# Historical vs Demographics-Only Comparison

The ablation experiments revealed a significant performance drop when historical political features were removed.

This confirms that:
- historical voting behavior contributes the majority of predictive power,
- while demographic and socioeconomic features provide secondary but meaningful explanatory capability.

The comparison also demonstrated that Random Forest models are more capable of learning demographic-only political relationships than linear models such as Logistic Regression.

---

# Final Conclusion

This project successfully demonstrated that county-level election outcomes can be predicted with high accuracy using machine learning techniques.

The study revealed that:
- historical political behavior is the strongest predictor of future election outcomes,
- but demographic and socioeconomic variables still retain substantial independent predictive power.

Additionally, the project showed that:
- education level,
- racial composition,
- and socioeconomic conditions

play major roles in shaping political alignment across U.S. counties.

Overall, the project achieved strong predictive performance while also producing meaningful and interpretable political insights through feature importance analysis and comparative modeling.