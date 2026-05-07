# Data Dictionary

## Project

Predicting U.S. County-Level Presidential Voting Outcomes Using Demographic and Socioeconomic Machine Learning Models

---

# Dataset Sources

## 1. MIT Election Dataset

Source: MIT Election Data and Science Lab

Purpose:
Provides county-level presidential election results used for classification and regression targets.

### Important Variables

| Variable       | Description                                |
| -------------- | ------------------------------------------ |
| state          | Full state name                            |
| state_po       | State abbreviation                         |
| county_name    | County name                                |
| county_fips    | County FIPS code used for merging datasets |
| year           | Election year                              |
| office         | Election office (US PRESIDENT)             |
| candidate      | Candidate name                             |
| party          | Political party                            |
| candidatevotes | Votes received by candidate                |
| totalvotes     | Total votes in county                      |

### Target Variables Created

| Variable       | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| dem_vote_share | Percentage of Democratic votes in county                    |
| party_winner   | Binary classification target (1 = Democrat, 0 = Republican) |

---

# ACS Demographic Datasets

## 2. Age and Sex Dataset — B01001

Purpose:
Provides county-level age and sex distribution data.

### Important Variables

| Variable    | Description                        |
| ----------- | ---------------------------------- |
| GEO_ID      | Census geographic identifier       |
| NAME        | County name                        |
| B01001_001E | Total population                   |
| B01001_*    | Age and sex distribution estimates |

---

## 3. Race Dataset — B02001

Purpose:
Provides racial composition information for each county.

### Important Variables

| Variable    | Description                          |
| ----------- | ------------------------------------ |
| B02001_001E | Total population                     |
| B02001_002E | White population                     |
| B02001_003E | Black or African American population |
| B02001_005E | Asian population                     |

---

## 4. Education Dataset — B15003

Purpose:
Provides educational attainment statistics.

### Important Variables

| Variable    | Description                        |
| ----------- | ---------------------------------- |
| B15003_001E | Total population 25 years and over |
| B15003_*    | Educational attainment categories  |

---

# ACS Socioeconomic Datasets

## 5. Income Dataset — B19013

Purpose:
Provides median household income estimates.

### Important Variables

| Variable    | Description             |
| ----------- | ----------------------- |
| B19013_001E | Median household income |

---

## 6. Poverty Dataset — B17001

Purpose:
Provides poverty status statistics.

### Important Variables

| Variable    | Description                                      |
| ----------- | ------------------------------------------------ |
| B17001_001E | Population for whom poverty status is determined |
| B17001_*    | Poverty status breakdown                         |

---

## 7. Employment Dataset — B23025

Purpose:
Provides employment and labor force statistics.

### Important Variables

| Variable    | Description                  |
| ----------- | ---------------------------- |
| B23025_001E | Population 16 years and over |
| B23025_002E | Labor force                  |
| B23025_004E | Employed population          |
| B23025_005E | Unemployed population        |

---

## 8. Housing Dataset — B25001

Purpose:
Provides housing unit estimates.

### Important Variables

| Variable    | Description         |
| ----------- | ------------------- |
| B25001_001E | Total housing units |

---

# Merge Key

All datasets will be merged using:

| Variable    | Description      |
| ----------- | ---------------- |
| county_fips | County FIPS code |

For ACS datasets, county FIPS will be extracted from:

* GEO_ID

Example:

* GEO_ID = 0500000US01001
* county_fips = 01001

---

# Modeling Tasks

## Classification

Predict:

* Democrat vs Republican county winner

### Models

* Logistic Regression
* Decision Tree
* Random Forest
* Boosting
* Naïve Bayes

### Evaluation

* Accuracy
* Precision
* Recall
* F1-score

---

## Regression

Predict:

* Democratic vote share

### Models

* Linear Regression
* Random Forest Regression
* Boosting
* Support Vector Regression

### Evaluation

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

## Clustering

Group counties based on demographic and socioeconomic characteristics.

### Methods

* K-Means Clustering
* Hierarchical Clustering

### Evaluation

* Silhouette Score
* Cluster interpretation
