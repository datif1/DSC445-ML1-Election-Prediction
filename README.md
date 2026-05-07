# DSC445-ML1-Election-Prediction

## Predicting U.S. County-Level Presidential Voting Outcomes Using Demographic and Socioeconomic Machine Learning Models

This repository contains the implementation and analysis for the Machine Learning 1 (DSC 445) final project focused on predicting U.S. county-level presidential voting outcomes using demographic and socioeconomic data.

---

# Team Members

* Veon Ivon Almeida
* Shang Andrews
* Mohammed Atif Diwan
* Yung Han Jeong
* Siddhesh Dayanand Patil

---

# Project Objective

The goal of this project is to analyze whether demographic and socioeconomic characteristics of U.S. counties can be used to predict presidential voting behavior.

The project focuses on:

* Predicting Democrat vs Republican county outcomes
* Predicting Democratic vote share percentage
* Identifying influential demographic and socioeconomic features
* Comparing multiple machine learning approaches
* Exploring county clustering patterns using unsupervised learning

---

# Datasets

## 1. MIT Election Dataset

Source:
MIT Election Data and Science Lab

Dataset:
County Presidential Election Returns (2000–2024)

Contains:

* county-level election results
* party information
* vote counts
* total votes
* county FIPS codes

---

## 2. ACS Demographic Datasets

Includes:

* Age and Sex (B01001)
* Race (B02001)
* Education (B15003)

---

## 3. ACS Socioeconomic Datasets

Includes:

* Income (B19013)
* Poverty (B17001)
* Employment (B23025)
* Housing (B25001)

All ACS datasets use 2020 ACS 5-Year Estimates at county level.

---

# Project Structure

```text
DSC445-ML1-Election-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── visualizations/
│
├── models/
│
├── reports/
│
├── docs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Machine Learning Tasks

## Classification

Predict:

* Democrat vs Republican county winner

Models:

* Logistic Regression
* Decision Tree
* Random Forest
* Boosting
* Naïve Bayes

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-score

---

## Regression

Predict:

* Democratic vote share

Models:

* Linear Regression
* Random Forest Regression
* Boosting
* Support Vector Regression

Evaluation Metrics:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

## Clustering

Group counties based on demographic and socioeconomic characteristics.

Methods:

* K-Means Clustering
* Hierarchical Clustering

Evaluation:

* Silhouette Score
* Cluster interpretation

---

# Data Processing

Datasets are merged using county FIPS codes.

Processing steps include:

* Data cleaning
* County FIPS extraction
* Feature engineering
* Missing value handling
* Dataset merging
* Feature scaling and preparation

---

# Current Status

* Project proposal finalized
* Datasets collected and organized
* Repository initialized
* Preprocessing and exploratory data analysis in progress

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

# Course

DSC 445 — Machine Learning 1
DePaul University
