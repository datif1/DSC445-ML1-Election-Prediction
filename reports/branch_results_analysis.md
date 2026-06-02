# DSC 445 Branch Results Analysis

## Project Context

This analysis summarizes the results from three project branches created for the DSC 445 Machine Learning I election prediction project:

- `feature/shang-classification`
- `feature/shang-regression`
- `feature/shang-clustering`

The project goal is to predict and analyze U.S. county-level presidential voting outcomes using county-level demographic and socioeconomic features. The dataset combines MIT Election Lab county-level presidential election results with ACS demographic, socioeconomic, education, race, employment, poverty, housing, and age variables.

The main value of these branches is that they move the project beyond raw exploratory analysis and into structured machine learning workflows. Together, the branches show that the dataset is usable, that the models can learn meaningful signal from county-level features, and that unsupervised clustering can identify interpretable county archetypes.

---

## Branch Overview

### 1. `feature/shang-classification`

This branch added the classification training pipeline and baseline classification results.

Major files added or changed:

- `src/classification_models.py`
- `reports/classification_model_results.csv`
- `reports/classification_feature_importance.csv`
- `Makefile`
- processed datasets
- preprocessing and EDA notebooks
- EDA report

Branch summary:

```text
10 files changed, 14,731 insertions, 1 deletion
```

Key commits:

```text
f0a5c07 Add Makefile for project workflow commands
f3522be Implement classification training pipeline and baseline results
014bced Add local virtual environment to gitignore
a2dddba Add processed datasets
c44d02b EDA report Update
6146c70 Add preprocessing and enhanced EDA work
```

This branch focuses on predicting the county-level winning party using `party_winner` as the target variable.

---

### 2. `feature/shang-regression`

This branch added the regression training pipeline and baseline regression results.

Major files added or changed:

- `src/regression_models.py`
- `reports/regression_model_results.csv`
- `reports/regression_feature_importance.csv`
- processed datasets
- preprocessing and EDA notebooks
- EDA report

Branch summary:

```text
9 files changed, 14,868 insertions
```

Key commits:

```text
70b15af Implement regression training pipeline and baseline results
a2dddba Add processed datasets
c44d02b EDA report Update
6146c70 Add preprocessing and enhanced EDA work
```

This branch focuses on predicting the actual Democratic vote share using `dem_vote_share` as the target variable.

---

### 3. `feature/shang-clustering`

This branch added unsupervised clustering and county archetype discovery.

Major files added or changed:

- `src/clustering.py`
- `reports/clustering_model_results.csv`
- `reports/clustering_cluster_summary.csv`
- `reports/clustering_county_assignments.csv`
- processed datasets
- preprocessing and EDA notebooks
- EDA report

Branch summary:

```text
10 files changed, 17,170 insertions
```

Key commits:

```text
cafc9c9 comment at the bottom explaining results
a1603c2 Implement clustering analysis and county archetype discovery
a2dddba Add processed datasets
c44d02b EDA report Update
6146c70 Add preprocessing and enhanced EDA work
```

This branch focuses on grouping counties into interpretable clusters based on demographic and socioeconomic patterns without directly training on the election target.

---

## Dataset Inspection Results

The model-ready dataset has:

```text
Dataset shape: (3115, 182)
```

This means the final model-ready dataset contains 3,115 county-level records and 182 columns after feature engineering.

The classification target distribution is:

```text
party_winner
0    2576
1     539
```

The normalized class distribution is:

```text
party_winner
0    0.826966
1    0.173034
```

This shows a major class imbalance. About 82.7% of counties were Republican-winning counties, while about 17.3% were Democrat-winning counties.

This is important because a model could achieve high accuracy simply by predicting the majority class too often. For that reason, classification performance should be judged using precision, recall, and F1-score in addition to accuracy.

The Democratic vote share distribution was:

```text
count    3115.000000
mean        0.332783
std         0.159654
min         0.030909
25%         0.209402
50%         0.299816
75%         0.424319
max         0.921497
```

This means the average county-level Democratic vote share was about 33.3%, while the median was about 30.0%. The distribution confirms that most counties leaned Republican, while a smaller number of counties had very high Democratic vote share.

Only two columns had missing values:

```text
median_income    1
B19013_001E      1
```

This is a very small amount of missing data relative to the full dataset. The model scripts handle missing values using median imputation, so the missing income value does not block modeling.

The following engineered features were confirmed to exist in the model-ready dataset:

```text
higher_education_rate    True
poverty_rate             True
unemployment_rate        True
white_pct                True
black_pct                True
asian_pct                True
log_population           True
bachelors_degree         True
masters_degree           True
professional_degree      True
doctorate_degree         True
```

These engineered variables are important because the EDA showed that raw count variables are heavily affected by county population size. Percentages, rates, and log-transformed population allow the models to compare counties more fairly.

---

## EDA Findings and Modeling Implications

The EDA established several important patterns that influenced the modeling strategy.

First, the dataset has strong class imbalance. Republican-winning counties greatly outnumber Democrat-winning counties. This does not contradict the national presidential result because Democratic voters are more concentrated in highly populated urban counties, while Republican-winning counties are more numerous geographically. For modeling, this means accuracy alone is not enough. Precision, recall, F1-score, and confusion matrix results are needed to judge classification performance.

Second, raw population counts are not always reliable indicators because larger counties naturally have higher counts for almost every demographic group. This can make raw poverty counts, raw education counts, and raw race counts misleading. The project addressed this issue by creating normalized features such as `higher_education_rate`, `poverty_rate`, `unemployment_rate`, `white_pct`, `black_pct`, `asian_pct`, and `log_population`.

Third, education-related variables emerged as especially important. The EDA report identified `higher_education_rate` as one of the strongest relationships with Democratic vote share. This finding was later supported by the classification and regression feature importance outputs, where higher education and advanced degree variables repeatedly appeared among the most influential model features.

Fourth, racial composition variables such as `white_pct`, `black_pct`, and `asian_pct` appeared repeatedly in the strongest tree-based feature importance results. These variables should be interpreted as statistical predictors in the dataset, not causal explanations. They show that county demographic composition is associated with voting outcomes in the data, but they do not by themselves explain why individual voters behave a certain way.

---

## Classification Results Analysis

The classification branch trained models to predict:

```text
party_winner
1 = Democrat
0 = Republican
```

The classification results were:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.9262 | 0.8039 | 0.7593 | 0.7810 |
| Logistic Regression | 0.9069 | 0.6712 | 0.9074 | 0.7717 |
| Random Forest | 0.9053 | 0.7634 | 0.6574 | 0.7065 |
| Decision Tree | 0.8732 | 0.6043 | 0.7778 | 0.6802 |
| Naive Bayes | 0.8587 | 0.6613 | 0.3796 | 0.4824 |

### Best Overall Classification Model

The strongest overall classification model was Gradient Boosting.

It had:

```text
accuracy  = 0.9262
precision = 0.8039
recall    = 0.7593
f1_score  = 0.7810
```

Gradient Boosting had the highest F1-score and the highest accuracy. Because F1-score balances precision and recall, this makes Gradient Boosting the strongest overall baseline model for predicting county winner.

The model was not only accurate overall, but also relatively balanced in how it handled the minority class. This matters because Democrat-winning counties were only about 17.3% of the dataset.

### Logistic Regression Interpretation

Logistic Regression had slightly lower F1-score than Gradient Boosting, but it had the highest recall:

```text
recall = 0.9074
```

This means Logistic Regression was very strong at identifying Democrat-winning counties. However, its precision was lower:

```text
precision = 0.6712
```

That means it likely classified more Republican counties as Democrat than Gradient Boosting did.

This tradeoff is important. If the goal is to catch as many Democrat-winning counties as possible, Logistic Regression is useful. If the goal is better balance between false positives and false negatives, Gradient Boosting is better.

### Random Forest Interpretation

Random Forest performed well but did not outperform Gradient Boosting.

```text
accuracy  = 0.9053
precision = 0.7634
recall    = 0.6574
f1_score  = 0.7065
```

Random Forest had decent precision, but its lower recall means it missed more Democrat-winning counties than Gradient Boosting and Logistic Regression. This suggests Random Forest was more conservative in predicting the minority class.

### Decision Tree Interpretation

The Decision Tree had:

```text
accuracy  = 0.8732
precision = 0.6043
recall    = 0.7778
f1_score  = 0.6802
```

The Decision Tree was easier to interpret but weaker overall. It caught a reasonable number of Democrat-winning counties, but with lower precision. This means the simpler tree structure likely overgeneralized some decision boundaries.

### Naive Bayes Interpretation

Naive Bayes had the weakest F1-score:

```text
f1_score = 0.4824
recall   = 0.3796
```

The low recall shows that Naive Bayes missed many Democrat-winning counties. This suggests the assumptions behind Naive Bayes did not fit this dataset well, especially given the correlated demographic and socioeconomic features.

### Classification Takeaway

The classification branch shows that county winner can be predicted with strong baseline performance from demographic and socioeconomic features.

The key finding is not just that accuracy is high. The key finding is that Gradient Boosting performs best when judged by F1-score, while Logistic Regression performs best when prioritizing recall for Democrat-winning counties.

Because the dataset is imbalanced, F1-score should be treated as the most useful single classification metric in the current results.

---

## Classification Feature Importance Analysis

The classification feature importance results show that the models relied heavily on education, race percentage, poverty, unemployment, and age/demographic variables.

### Logistic Regression

Top Logistic Regression features included:

```text
B01001_024E
B02001_009E
B15003_021E
higher_education_rate
B15003_014E
B02001_008E
B01001_044E
B01001_020E
B23025_007E
B17001_050E
B01001_029E
B15003_018E
```

The most interpretable result here is that `higher_education_rate` was one of the strongest Logistic Regression features. Some of the top raw ACS-coded features would require a data dictionary to label precisely, but the appearance of education and demographic ACS variables is consistent with the EDA.

### Decision Tree

Top Decision Tree features included:

```text
doctorate_degree
white_pct
higher_education_rate
B17001_023E
poverty_rate
B17001_037E
unemployment_rate
professional_degree
B15003_025E
B17001_045E
asian_pct
B17001_005E
```

This result is highly interpretable. The Decision Tree relied heavily on advanced education, racial composition, poverty, and unemployment. Its top three features were:

```text
doctorate_degree
white_pct
higher_education_rate
```

This supports the EDA finding that education and demographic composition are strongly related to county-level voting outcomes.

### Random Forest

Top Random Forest features included:

```text
white_pct
higher_education_rate
black_pct
B15003_025E
doctorate_degree
B02001_005E
poverty_rate
asian_population
B15003_023E
B15003_024E
B02001_003E
black_population
```

Random Forest distributed importance more broadly than the single Decision Tree. This makes sense because Random Forest averages many trees. Still, the same pattern appears: race percentage, higher education, poverty, and advanced education variables are repeatedly important.

### Gradient Boosting

Top Gradient Boosting features included:

```text
white_pct
higher_education_rate
B01001_033E
doctorate_degree
B15003_025E
B17001_024E
black_pct
asian_pct
poverty_rate
unemployment_rate
B02001_007E
B01001_031E
```

Gradient Boosting was the best classification model, so its feature importance is especially useful. Its strongest features were:

```text
white_pct
higher_education_rate
doctorate_degree
black_pct
asian_pct
poverty_rate
unemployment_rate
```

This confirms that the best-performing classifier relied heavily on normalized and interpretable features, not just raw counts.

### Classification Feature Importance Takeaway

Across the classification models, the most consistent predictors were:

- `higher_education_rate`
- `white_pct`
- `black_pct`
- `asian_pct`
- `poverty_rate`
- `unemployment_rate`
- advanced degree variables such as `doctorate_degree` and `professional_degree`

The repeated appearance of these features across different model types increases confidence that these are not random artifacts from one model. They appear to represent meaningful county-level patterns in the dataset.

---

## Regression Results Analysis

The regression branch trained models to predict:

```text
dem_vote_share
```

This is a more detailed target than `party_winner` because it predicts the level of Democratic support instead of only predicting which party won the county.

The regression results were:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Support Vector Regressor | 0.0581 | 0.0059 | 0.0766 | 0.7615 |
| Gradient Boosting Regressor | 0.0604 | 0.0063 | 0.0796 | 0.7427 |
| Random Forest Regressor | 0.0613 | 0.0067 | 0.0816 | 0.7293 |
| Decision Tree Regressor | 0.0781 | 0.0103 | 0.1015 | 0.5812 |
| Linear Regression | 0.0753 | 0.0156 | 0.1247 | 0.3679 |
| Ridge Regression | 0.0752 | 0.0183 | 0.1354 | 0.2555 |

### Best Overall Regression Model

The best regression model was Support Vector Regressor.

It had:

```text
MAE      = 0.0581
RMSE     = 0.0766
R² score = 0.7615
```

The MAE of 0.0581 means the model was off by about 5.8 percentage points in Democratic vote share on average. For a county-level political prediction task, this is a strong baseline result.

The R² score of about 0.7615 means the model explained about 76.1% of the variation in Democratic vote share in the test set.

### Gradient Boosting and Random Forest Regression

Gradient Boosting Regressor and Random Forest Regressor were close behind:

```text
Gradient Boosting R² = 0.7427
Random Forest R²     = 0.7293
```

These are also strong results. The fact that Support Vector Regression, Gradient Boosting, and Random Forest all performed well suggests the engineered county features contain substantial predictive signal.

### Decision Tree Regression

The Decision Tree Regressor had:

```text
R² = 0.5812
```

This is weaker than the ensemble and SVR models. A single tree likely cannot capture enough complexity in the data without overfitting or oversimplifying the relationships.

### Linear and Ridge Regression

Linear Regression and Ridge Regression performed much worse:

```text
Linear Regression R² = 0.3679
Ridge Regression R²  = 0.2555
```

This suggests that the relationship between county features and Democratic vote share is not purely linear. The stronger performance of SVR, Gradient Boosting, and Random Forest indicates that nonlinear modeling is more appropriate for this dataset.

### Regression Takeaway

The regression branch shows that Democratic vote share can be predicted with meaningful accuracy from county-level features. The best model, Support Vector Regressor, produced an average error of about 5.8 percentage points and explained about 76% of the variation in the target.

This is one of the strongest results in the project because it shows the model can estimate not only which party won a county, but also the intensity of Democratic support.

---

## Regression Feature Importance Analysis

Support Vector Regressor was the best regression model, but the current feature importance CSV does not include SVR because standard RBF SVR does not expose simple feature importance values. Therefore, the most useful regression feature interpretation comes from the tree-based and linear models.

### Decision Tree Regressor

Top Decision Tree Regressor features included:

```text
doctorate_degree
white_pct
B15003_024E
higher_education_rate
B15003_025E
B23025_007E
poverty_rate
B15003_011E
B17001_037E
masters_degree
B17001_009E
B02001_005E
```

The top features were highly concentrated:

```text
doctorate_degree importance = 0.3850
white_pct importance         = 0.3007
higher_education_rate        = 0.0982
```

This shows that the single regression tree relied heavily on advanced education and racial composition.

### Random Forest Regressor

Top Random Forest Regressor features included:

```text
white_pct
doctorate_degree
B15003_025E
higher_education_rate
B02001_005E
asian_population
asian_pct
B15003_024E
professional_degree
black_pct
poverty_rate
B02001_003E
```

This again supports the importance of racial composition, higher education, advanced degrees, and poverty.

### Gradient Boosting Regressor

Top Gradient Boosting Regressor features included:

```text
white_pct
higher_education_rate
doctorate_degree
B15003_025E
asian_pct
B15003_024E
black_pct
B17001_024E
poverty_rate
professional_degree
B15003_021E
B02001_009E
```

The best interpretable regression model was Gradient Boosting Regressor. Its strongest features were:

```text
white_pct importance                = 0.2887
higher_education_rate importance    = 0.1748
doctorate_degree importance         = 0.1104
asian_pct importance                = 0.0628
black_pct importance                = 0.0260
poverty_rate importance             = 0.0148
```

These results align strongly with the classification feature importance outputs.

### Regression Feature Importance Takeaway

The regression feature importance results show that the same types of features explain both county winner and Democratic vote share:

- racial composition percentages
- higher education rate
- advanced degree counts/rates
- poverty rate
- employment-related variables
- Asian and Black population percentage variables

The consistency between classification and regression results strengthens the overall project finding: county-level demographic and socioeconomic features contain meaningful predictive signal for both categorical and continuous election outcomes.

---

## Clustering Results Analysis

The clustering branch used unsupervised learning to group counties into archetypes. Unlike classification and regression, clustering does not directly predict the election target. Instead, it groups counties based on feature similarity and then examines how those clusters relate to voting outcomes.

### K-Means Model Selection

The clustering script evaluated K-Means cluster counts from 2 through 10 and compared silhouette scores and inertia.

The results were:

| Model | Clusters | Silhouette Score | Inertia |
|---|---:|---:|---:|
| KMeans | 2 | 0.3331 | 18988.8186 |
| KMeans | 3 | 0.3431 | 14392.4371 |
| KMeans | 4 | 0.2166 | 12507.2436 |
| KMeans | 5 | 0.2253 | 10939.4545 |
| KMeans | 6 | 0.1923 | 10112.0368 |
| KMeans | 7 | 0.1919 | 9403.9313 |
| KMeans | 8 | 0.1960 | 8823.9694 |
| KMeans | 9 | 0.1971 | 8260.0910 |
| KMeans | 10 | 0.1868 | 7841.2445 |
| Agglomerative Clustering | 3 | 0.2954 | N/A |

The best K-Means result by silhouette score was:

```text
K = 3
silhouette_score = 0.3431
```

This suggests that the strongest clustering structure found by K-Means was a 3-cluster solution.

Agglomerative Clustering with 3 clusters had a lower silhouette score:

```text
silhouette_score = 0.2954
```

This means K-Means produced the better cluster separation according to the selected metric.

### Cluster Summary

The three K-Means clusters were:

| Cluster | Counties | Republican Counties | Democrat Counties | Avg Dem Vote Share | Democrat County Share |
|---|---:|---:|---:|---:|---:|
| Cluster 0 | 2067 | 1989 | 78 | 0.2651 | 0.0377 |
| Cluster 1 | 506 | 223 | 283 | 0.5120 | 0.5593 |
| Cluster 2 | 542 | 364 | 178 | 0.4236 | 0.3284 |

### Cluster 0: Rural Republican Counties

Cluster 0 contained:

```text
2067 counties
1989 Republican-winning counties
78 Democrat-winning counties
average Democratic vote share = 0.2651
Democrat county share = 0.0377
```

This cluster represents the largest group of counties in the dataset. It was overwhelmingly Republican-winning and had a low average Democratic vote share.

Its average characteristics were:

```text
avg_total_population        = 35,826
avg_median_income           = 53,250
avg_higher_education_rate   = 0.1419
avg_poverty_rate            = 0.1307
avg_unemployment_rate       = 0.0452
avg_white_pct               = 0.8969
avg_black_pct               = 0.0313
avg_asian_pct               = 0.0071
```

This supports the label “Rural Republican Counties.” These counties were smaller, less diverse, less highly educated on average, and strongly Republican in county winner outcomes.

### Cluster 1: Urban / Diverse / Highly Educated Counties

Cluster 1 contained:

```text
506 counties
223 Republican-winning counties
283 Democrat-winning counties
average Democratic vote share = 0.5120
Democrat county share = 0.5593
```

This was the only cluster where Democrat-winning counties formed the majority.

Its average characteristics were:

```text
avg_total_population        = 433,868
avg_median_income           = 76,132
avg_higher_education_rate   = 0.2621
avg_poverty_rate            = 0.0979
avg_unemployment_rate       = 0.0463
avg_white_pct               = 0.7678
avg_black_pct               = 0.0940
avg_asian_pct               = 0.0483
```

This supports the label “Urban / Diverse / Highly Educated Counties.” These counties had the highest average population, highest median income, highest higher education rate, highest Asian percentage, and the highest Democratic vote share.

This cluster is important because it captures the group of counties where Democratic support is most concentrated.

### Cluster 2: Competitive / Transitional Counties

Cluster 2 contained:

```text
542 counties
364 Republican-winning counties
178 Democrat-winning counties
average Democratic vote share = 0.4236
Democrat county share = 0.3284
```

This cluster was still majority Republican-winning, but much more competitive than Cluster 0.

Its average characteristics were:

```text
avg_total_population        = 60,043
avg_median_income           = 41,467
avg_higher_education_rate   = 0.1125
avg_poverty_rate            = 0.2181
avg_unemployment_rate       = 0.0794
avg_white_pct               = 0.5777
avg_black_pct               = 0.3152
avg_asian_pct               = 0.0077
```

This supports the label “Competitive / Transitional Counties.” These counties had higher poverty, higher unemployment, lower median income, lower higher education rates, and higher Black population share than the other clusters. Politically, they sat between Cluster 0 and Cluster 1 in Democratic vote share.

### PCA Interpretation

The clustering script included PCA coordinates for visualization.

The PCA summary was:

```text
PC1 = 40.0%
PC2 = 29.3%
Total = 69.3%
```

This means the first two principal components explain about 69.3% of the variation in the scaled clustering feature space. That is strong enough to support a useful two-dimensional visualization of the county clusters.

### Clustering Takeaway

The clustering branch successfully identified three interpretable county archetypes:

1. Rural Republican Counties
2. Urban / Diverse / Highly Educated Counties
3. Competitive / Transitional Counties

The clusters align well with the supervised learning results. Education, racial composition, income, population size, poverty, and unemployment help distinguish county groups, and those groups also have meaningfully different voting outcomes.

This makes the clustering branch valuable because it provides an unsupervised explanation layer. It does not just predict an outcome; it helps describe the structure of the counties in the dataset.

---

## Cross-Branch Findings

The three branches support each other.

### Finding 1: The dataset is viable for machine learning

The processed dataset has 3,115 rows, 182 columns, and only one missing value in the income-related fields. This is clean enough for baseline modeling and analysis.

### Finding 2: The classification task is feasible

Gradient Boosting achieved the best F1-score:

```text
F1-score = 0.7810
```

This is a strong baseline given the major class imbalance.

### Finding 3: The regression task gives the most nuanced predictive result

Support Vector Regressor achieved:

```text
MAE = 0.0581
R²  = 0.7615
```

This means the model can predict Democratic vote share with an average error of about 5.8 percentage points.

### Finding 4: Nonlinear models perform better

In both classification and regression, nonlinear models performed better than simpler linear approaches.

Classification:

- Gradient Boosting had the best F1-score.
- Logistic Regression had strong recall but lower precision.

Regression:

- Support Vector Regressor, Gradient Boosting Regressor, and Random Forest Regressor were much stronger than Linear Regression and Ridge Regression.

This suggests the relationship between county features and voting outcomes is nonlinear.

### Finding 5: Education and demographic composition are consistently important

Across EDA, classification, regression, and clustering, the most important recurring variables were:

- higher education rate
- advanced degree variables
- white population percentage
- Black population percentage
- Asian population percentage
- poverty rate
- unemployment rate
- income
- population size / log population

This consistency increases confidence that the models are finding stable patterns in the data.

### Finding 6: Clustering adds interpretability

The clustering branch gives a useful way to describe counties as groups rather than only predicting individual outcomes. The three-cluster solution provides a clean story for the project:

- most counties fall into a rural Republican-leaning cluster,
- a smaller set of larger and more educated counties leans Democratic,
- an intermediate group contains more competitive counties with distinct socioeconomic patterns.

---

## Recommended Merge Plan

A reasonable merge plan would be:

1. Merge the shared preprocessing, processed dataset, and EDA work first.
2. Merge `feature/shang-classification`.
3. Merge `feature/shang-regression`.
4. Merge `feature/shang-clustering`.
5. Use `development-staging` or `shang-integration` as the combined testing branch before merging into the final main branch.

The reason for this order is that the classification, regression, and clustering branches all depend on the processed model-ready dataset. The data pipeline and EDA foundation should be treated as the base layer. Then the supervised and unsupervised modeling branches can be merged on top of it.

The `feature/shang-classification` branch also adds a `Makefile`, which can help standardize commands such as:

```bash
make classify
make regress
make cluster
```

However, before final merging, the team should confirm that the Makefile works consistently on each teammate’s environment, especially because the `clean` command currently uses Windows-style deletion syntax.

---

## Limitations

There are several limitations to mention clearly.

First, the dataset is county-level, not individual-level. The models show relationships between county characteristics and county outcomes, but they should not be interpreted as explaining individual voter behavior.

Second, the dataset has a strong class imbalance. Republican-winning counties are much more common than Democrat-winning counties. This makes F1-score, precision, and recall more important than accuracy for classification.

Third, some feature importance outputs still include raw ACS column names such as `B01001_033E` or `B17001_024E`. These should be mapped to readable labels using the ACS data dictionary before the final presentation or report.

Fourth, feature importance is not the same as causation. Variables such as education rate, race percentage, poverty rate, and unemployment rate are predictive in the dataset, but the analysis should not claim that any one variable directly causes voting outcomes.

Fifth, the regression feature importance results do not explain the Support Vector Regressor directly because the RBF SVR model does not expose simple feature importances. The feature interpretation therefore comes mainly from Gradient Boosting, Random Forest, Decision Tree, Linear Regression, and Ridge Regression.

Sixth, clustering labels are interpretations created after examining cluster summaries. The algorithm only produces groups; the names “Rural Republican,” “Urban / Diverse / Highly Educated,” and “Competitive / Transitional” are human-readable labels based on the average characteristics of each cluster.

---

## Recommended Next Steps

### 1. Create readable feature labels

Map ACS-coded variables to readable names. This will make the report and presentation much easier to understand.

For example:

```text
B15003_025E -> Doctorate degree
B15003_024E -> Professional degree
B23025_007E -> Not in labor force
```

The exact mappings should be verified against the ACS table documentation.

### 2. Add confusion matrices

The classification results should include confusion matrices for at least:

- Gradient Boosting
- Logistic Regression
- Random Forest

This would make it easier to explain false positives and false negatives.

### 3. Add visualizations

Useful visuals would include:

- bar chart of classification F1-scores
- bar chart of regression R² scores
- feature importance bar chart for Gradient Boosting
- PCA cluster scatterplot
- boxplot of Democratic vote share by cluster
- class distribution chart for `party_winner`

### 4. Compare classification and regression conclusions

The final report should explain that classification answers:

```text
Can we predict which party wins a county?
```

Regression answers:

```text
Can we predict the level of Democratic support in a county?
```

Clustering answers:

```text
What kinds of counties exist in the dataset based on demographic and socioeconomic structure?
```

Together, these three modeling approaches make the project stronger than using only one model type.

### 5. Use clustering as an interpretation layer

The clustering results can help explain model behavior. For example, a county in Cluster 1 is more likely to have higher Democratic vote share, while a county in Cluster 0 is much more likely to be Republican-winning. This can be used to explain the broader structure behind the supervised model predictions.

---

## Final Conclusion

The three branches successfully move the project from data preparation into meaningful machine learning analysis.

The classification branch shows that county-level winner prediction is feasible, with Gradient Boosting producing the strongest balanced classification performance.

The regression branch shows that Democratic vote share can be predicted with strong accuracy, with Support Vector Regressor achieving the best overall regression performance and an average error of about 5.8 percentage points.

The clustering branch adds interpretability by identifying three county archetypes that align with meaningful differences in population size, income, education, racial composition, poverty, unemployment, and Democratic vote share.

The strongest overall project conclusion is that engineered demographic and socioeconomic county-level features contain real predictive signal for election outcomes. However, the project should carefully frame the results as county-level statistical patterns, not individual voter explanations or causal claims.
