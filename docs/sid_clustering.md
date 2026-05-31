# Sid — Clustering

**Notebook:** `sid_clustering.ipynb`  
**Branch:** `siddhesh`  
**Input:** `../data/processed/model_ready_dataset.csv`

---

## Overview

K-Means clustering applied to discover natural county groupings based purely on demographic features — without using any election labels.

---

## Features Used

```
higher_ed_rate, poverty_rate, unemployment_rate,
white_pct, black_pct, asian_pct,
median_income, log_population, housing_density
```

---

## Optimal k Selection

| k | Silhouette Score | Decision |
|---|-----------------|----------|
| 2 | 0.3015 | Good but too broad |
| 3 | **0.3094** | **OPTIMAL** |
| 4 | 0.1968 | Sharp drop |
| 5 | 0.2097 | Below k=3 |
| 6-10 | 0.18-0.20 | Consistently lower |

Elbow method also confirms k=3 as the bend point.

---

## Cluster Profiles

| Cluster | Profile | Avg Dem Share | Majority Winner |
|---------|---------|---------------|-----------------|
| 0 | Urban/Diverse — high education, high Asian% | 0.50 | Democrat (slight) |
| 1 | Mixed/Swing — high Black%, moderate income | 0.42 | Republican (slight) |
| 2 | Rural/White — high white%, low education | 0.26 | Republican (strong) |

---

## PCA Visualization

- 2 PCA components explain **63.5%** of total variance
- Clusters are visually well-separated in PCA space
- Cluster 2 (Rural/White) is the largest group (~2,054 counties)

---

## Key Finding

K-Means discovered politically-aligned county groups **without ever seeing election data**.  
Demographics alone are sufficient to separate counties by voting behavior — confirming that race and education are the dominant structural drivers of U.S. county-level voting patterns.
