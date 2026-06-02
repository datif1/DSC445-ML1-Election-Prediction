# Predicting U.S. County-Level Presidential Election Outcomes Using Demographic, Socioeconomic, and Historical Voting Data

## DSC 445 – Machine Learning 1 Final Project

### Team Members

* Mohammed Atif Diwan
* Shang Andrews
* Veon Ivon Almeida
* Yung Han Jeong
* Siddhesh Dayanand Patil

---

# 1. Introduction

The objective of this project was to investigate whether county-level demographic, socioeconomic, and historical voting information can be used to predict future presidential election outcomes in the United States.

Political behavior is influenced by a combination of long-term voting patterns, demographic composition, educational attainment, economic conditions, and population characteristics. By combining election data from the MIT Election Lab with demographic and socioeconomic information from the American Community Survey (ACS), this project aimed to develop machine learning models capable of forecasting county-level election outcomes.

The project explored three major machine learning tasks:

1. Classification of county-level election winners.
2. Regression prediction of Democratic vote share.
3. Unsupervised clustering of counties based on demographic characteristics.

In addition, ablation experiments were performed to measure the contribution of historical political information relative to demographic variables alone.

---

# 2. Data Collection and Integration

Two primary datasets were used:

### MIT Election Lab Dataset

The election dataset contained county-level presidential election results from 2000 through 2024.

Important variables included:

* Democratic vote totals
* Republican vote totals
* County-level election winners
* Democratic vote share

### American Community Survey Dataset

The ACS dataset provided county-level demographic and socioeconomic information including:

* Population
* Race composition
* Educational attainment
* Poverty statistics
* Employment information
* Housing characteristics
* Median household income

The datasets were merged using county FIPS codes.

The final merged dataset contained:

* 3,115 counties
* 162 original features
* Only one missing value

This provided a comprehensive county-level dataset covering both political and demographic characteristics.

---

# 3. Exploratory Data Analysis

Exploratory analysis revealed several important patterns.

### Class Imbalance

The county-level election winner variable showed a substantial imbalance.

* Republican-winning counties: 2,576
* Democrat-winning counties: 539

Although Democrats often receive large vote totals in highly populated urban counties, Republicans win a larger number of counties overall.

Because of this imbalance, model evaluation relied on precision, recall, F1-score, ROC-AUC, and confusion matrices rather than accuracy alone.

### Population Distribution

County populations were highly skewed.

A small number of counties contained extremely large populations, while most counties were relatively small.

This motivated the use of normalized percentage-based features rather than raw counts.

### Income and Education

Initial visualizations suggested that counties with higher income and education levels tended to exhibit stronger Democratic voting patterns.

### Poverty and Unemployment

Raw counts were difficult to interpret because larger counties naturally contain larger numbers of unemployed and impoverished individuals.

This motivated the feature engineering stage.

---

# 4. Feature Engineering

To improve interpretability and reduce population-size bias, several engineered variables were created.

### Demographic Rates

* Higher Education Rate
* Poverty Rate
* Unemployment Rate

### Race Composition Percentages

* White Percentage
* Black Percentage
* Asian Percentage

### Population Transformation

* Log Population

### Historical Political Features

To capture long-term voting behavior, several historical features were engineered:

* Democratic vote share from 2000–2020
* County winner from 2000–2020
* Average Democratic vote share
* Number of Democratic victories
* Voting volatility
* Election swing measures

These engineered variables became some of the strongest predictors throughout the project.

---

# 5. Classification Modeling

The first supervised learning task focused on predicting the 2024 county-level election winner.

Five classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Naive Bayes

### Classification Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 96.90%   |
| Decision Tree       | 97.88%   |
| Random Forest       | 97.23%   |
| XGBoost             | 99.18%   |
| Naive Bayes         | 89.56%   |

XGBoost achieved the strongest overall performance with:

* Accuracy: 99.18%
* Precision: 94.19%
* Recall: 100.00%
* F1 Score: 97.01%
* ROC-AUC: 99.90%

The model successfully identified nearly all county-level election outcomes.

---

# 6. Ablation Study

To measure the contribution of historical political information, demographics-only versions of all classification models were trained.

Historical voting variables were removed, leaving only ACS demographic and socioeconomic variables.

### Demographics-Only Results

The strongest demographics-only model was XGBoost with:

* Accuracy: 94.13%
* ROC-AUC: 96.63%

Although performance declined compared to the full-feature models, prediction accuracy remained high.

This demonstrates that demographics alone contain substantial predictive information.

However, historical voting behavior clearly provided the largest contribution to predictive performance.

---

# 7. Classification Insights

Several important insights emerged:

### Historical Voting Behavior Dominates Prediction

The most influential variables included:

* winner_2020
* winner_2016
* dem_share_2020
* dem_share_2016
* dem_wins_count_2000_2020

These features consistently appeared among the strongest predictors.

### Education is the Strongest Demographic Predictor

Higher education variables repeatedly emerged as the most influential demographic features.

Counties with higher rates of advanced educational attainment exhibited stronger Democratic voting tendencies.

### Race Composition Matters

White, Black, and Asian population percentages were consistently important predictors.

---

# 8. Regression Modeling

The second supervised learning task predicted the exact Democratic vote share for each county.

Five regression algorithms were evaluated:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regressor
* XGBoost Regressor

### Regression Results

| Model                   | R²     |
| ----------------------- | ------ |
| Linear Regression       | 0.7008 |
| Ridge Regression        | 0.7171 |
| Lasso Regression        | 0.7172 |
| Random Forest Regressor | 0.7491 |
| XGBoost Regressor       | 0.7760 |

XGBoost Regressor achieved the best overall performance.

Performance:

* MAE: 0.0590
* RMSE: 0.0879
* R²: 0.7760

---

# 9. Regression Insights

Historical voting variables once again dominated feature importance rankings.

The strongest predictors included:

* dem_share_2016
* dem_share_2020
* dem_wins_count_2000_2020
* avg_dem_share_2000_2020

These results reinforce the classification findings and demonstrate the long-term stability of county-level voting behavior.

---

# 10. Clustering Analysis

Unsupervised learning was performed to identify natural county groupings.

Two clustering techniques were explored:

* K-Means Clustering
* Hierarchical Clustering

### Initial Experiment

K-Means was first applied using the full feature set.

The resulting clusters were highly imbalanced and difficult to interpret.

### Feature Selection

A reduced feature set was created using:

* Higher Education Rate
* Poverty Rate
* White Percentage
* Black Percentage
* Asian Percentage

### Improved Clustering Results

Both K-Means and Hierarchical Clustering produced meaningful county segments including:

* Predominantly White Republican counties
* Working-class Republican counties
* Competitive counties
* Black-majority Democratic counties
* Highly educated urban counties

The consistency between the two clustering approaches strengthened confidence in the validity of the discovered groups.

---

# 11. Major Findings

The project produced several important findings:

1. Historical voting behavior is the strongest predictor of future election outcomes.

2. County-level political behavior exhibits substantial long-term stability.

3. Education level is the most influential demographic predictor.

4. Race composition significantly influences political alignment.

5. Demographic variables alone retain substantial predictive power even when historical political information is removed.

6. Tree-based ensemble methods consistently outperform linear models.

7. County demographic characteristics naturally form meaningful clusters that correspond to distinct political profiles.

---

# 12. Conclusion

This project successfully demonstrated that machine learning models can predict county-level presidential election outcomes with high accuracy.

Historical election results provided the greatest predictive power, while demographic and socioeconomic variables contributed meaningful additional information. Classification models achieved accuracy exceeding 99%, while regression models explained more than 77% of the variation in county-level Democratic vote share.

The findings reveal strong relationships between political behavior, educational attainment, race composition, and socioeconomic conditions. Furthermore, clustering analysis identified meaningful county segments that align closely with observed political patterns.

Overall, the project highlights the effectiveness of machine learning for analyzing and forecasting political behavior at the county level while providing interpretable insights into the demographic and historical factors that shape election outcomes.
