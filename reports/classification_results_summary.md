# Classification Model Results Summary

## Objective

The objective of the classification phase was to predict the winner of the 2024 U.S. Presidential Election at the county level using demographic, socioeconomic, and historical election features. Multiple machine learning algorithms were evaluated and compared to determine which approach produced the most accurate predictions.

---

## Models Evaluated

### Full Feature Models

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. Naive Bayes

### Demographics-Only Models

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. Naive Bayes

The demographics-only models excluded all historical voting variables and relied solely on ACS demographic and socioeconomic information.

---

## Classification Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 96.90% | 81.00% | 100.00% | 89.50% | 99.68% |
| Decision Tree | 97.88% | 86.96% | 98.77% | 92.49% | 99.86% |
| Random Forest | 97.23% | 83.33% | 98.77% | 90.40% | 99.72% |
| XGBoost | 99.18% | 94.19% | 100.00% | 97.01% | 99.90% |
| Naive Bayes | 89.56% | 77.42% | 29.63% | 42.86% | 81.30% |
| Demo-Only Logistic Regression | 87.77% | 52.27% | 85.19% | 64.79% | 94.13% |
| Demo-Only Decision Tree | 87.77% | 52.24% | 86.42% | 65.12% | 87.84% |
| Demo-Only Random Forest | 93.64% | 75.61% | 76.54% | 76.07% | 96.36% |
| Demo-Only XGBoost | 94.13% | 84.62% | 67.90% | 75.34% | 96.63% |
| Demo-Only Naive Bayes | 89.56% | 77.42% | 29.63% | 42.86% | 80.12% |

---

## Key Findings

### Best Overall Model

XGBoost achieved the strongest overall performance among all classification models.

Key metrics:

- Accuracy: 99.18%
- Precision: 94.19%
- Recall: 100.00%
- F1 Score: 97.01%
- ROC-AUC: 99.90%

The model correctly identified all Democratic-winning counties in the test set while producing only a small number of false positives.

---

### Impact of Historical Voting Features

Historical voting information substantially improved predictive performance.

For example:

- Full XGBoost Accuracy: 99.18%
- Demographics-Only XGBoost Accuracy: 94.13%

This decrease demonstrates that previous election outcomes and voting patterns provide important information beyond demographics alone.

The same trend was observed across Logistic Regression, Decision Tree, and Random Forest models.

---

### Importance of Demographic Variables

When historical voting information was removed, the most influential predictors included:

- Higher Education Rate
- White Population Percentage
- Black Population Percentage
- Asian Population Percentage
- Advanced Degree Attainment
- Median Household Income

These findings are consistent with patterns observed during exploratory data analysis.

---

### Naive Bayes Performance

Naive Bayes produced the weakest overall performance.

Although its accuracy remained relatively high, the model achieved very low recall and F1-score for Democratic-winning counties. This suggests that the independence assumptions of Naive Bayes do not adequately capture the complex relationships among demographic and voting variables.

---

## Overall Conclusion

The classification results demonstrate that county-level election outcomes can be predicted with very high accuracy using machine learning techniques.

Historical voting behavior emerged as the strongest predictor of future election outcomes. However, demographic and socioeconomic variables alone still provided substantial predictive power, achieving accuracy above 94% when used with XGBoost.

Among all evaluated models, XGBoost provided the most accurate and reliable predictions and was selected as the best-performing classification model in this study.