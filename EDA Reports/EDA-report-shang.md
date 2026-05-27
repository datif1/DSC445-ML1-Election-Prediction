# Exploratory Data Analysis Report

## DSC 445 – Machine Learning 1

### Predicting U.S. County-Level Presidential Voting Outcomes Using Demographic and Socioeconomic Features

#### Team Members
- Mohammed Atif Diwan
- Shang Andrews
- Veon Ivon Almeida
- Yung Han Jeong
- Siddhesh Dayanand Patil

---

# Exploratory Data Analysis (EDA) Summary

The exploratory data analysis for this project was conducted in two major phases. The first phase focused on analyzing the raw merged dataset created using the ACS demographic and socioeconomic datasets together with the MIT Election Lab county-level presidential election results. The second phase focused on improving the dataset through feature engineering and enhanced exploratory analysis using normalized and more interpretable variables.

The overall objective of the EDA process was to:
- understand the structure and quality of the data,
- identify relationships between county characteristics and voting outcomes,
- detect class imbalance and skewed features,
- create meaningful features for machine learning models,
- and prepare the dataset for classification and regression tasks.

---

# 1. Raw Dataset EDA

After preprocessing and merging all datasets using county FIPS codes, the final merged dataset contained:

- **3115 county-level records**
- **162 total features**
- only **1 missing value** in the median income column

This indicated that the preprocessing pipeline was highly successful and that the selected datasets were compatible for county-level analysis.

---

## Key Findings from the Raw Dataset

### A. Strong Class Imbalance

The target variable `party_winner` revealed a major class imbalance:

- Republican-winning counties: **2576**
- Democrat-winning counties: **539**

### Important Insight

Although Democrats won the 2020 presidential election nationally, Republicans won a much larger number of counties overall. This occurs because:
- Democratic votes are concentrated in highly populated urban counties
- Republican support is spread across many smaller rural counties

### Modeling Implication

Because of this imbalance:
- accuracy alone would not be reliable,
- evaluation metrics such as:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix  
must be used during modeling.

---

### B. Democratic Vote Share Distribution

The Democratic vote share distribution showed:
- most counties had lower Democratic vote percentages,
- while a smaller number of counties had very high Democratic support.

### Important Insight

This suggests:
- the dataset realistically captures political polarization,
- heavily Democratic counties form a smaller but politically concentrated group.

---

### C. Extreme Population Skewness

The county population distribution revealed:
- most counties have relatively small populations,
- while a few counties contain extremely large populations.

### Important Insight

Raw population-based variables were heavily influenced by county size.

This became an important limitation because:
- larger counties naturally contain larger counts for almost every demographic category,
- making raw-count comparisons potentially misleading.

---

### D. Median Income Differences

The boxplot comparing median household income by county winner showed:

- Democrat-winning counties generally had higher median household incomes
- Republican-winning counties tended to cluster at lower income ranges

### Important Insight

Income appeared to have a positive relationship with Democratic voting patterns, although there was still significant overlap between the two groups.

---

### E. Poverty and Unemployment Raw Counts

Initial poverty and unemployment analyses used raw counts.

The results showed:
- Democrat-winning counties had larger poverty-related populations,
- but interpretation was difficult because larger counties naturally have more people.

### Important Limitation

Raw counts did not provide fair county-to-county comparisons.

This directly motivated the need for:
- normalized features,
- percentage-based variables,
- and feature engineering.

---

### F. Initial Correlation Analysis

The initial correlation analysis identified strong relationships between:
- educational attainment variables,
- population-related variables,
- and Democratic vote share.

However, many correlations were dominated by county size effects rather than meaningful demographic relationships.

---

# 2. Enhanced Dataset EDA (Feature Engineering Phase)

To improve interpretability and reduce population-size bias, several engineered features were created.

Instead of raw counts, normalized variables were introduced, including:

- Higher Education Rate
- Poverty Rate
- Unemployment Rate
- White Population Percentage
- Black Population Percentage
- Asian Population Percentage
- Log-Transformed Population

This significantly improved the analytical quality of the dataset.

---

# 3. Key Findings from the Enhanced EDA

---

## A. Higher Education Rate

This became one of the strongest relationships in the project.

### Observations
- Democrat-winning counties showed significantly higher education rates
- Republican-winning counties clustered at lower education levels

### Correlation
```text
higher_education_rate ↔ dem_vote_share = +0.51