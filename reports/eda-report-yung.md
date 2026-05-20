# EDA Report: US Presidential Election — County-Level Prediction
**Analysis Year:** 2020  
**Prediction Tasks:** Binary Classification (Winner) · Regression (Democrat Vote Share) · Regression (Republican Vote Share)

---

## 1. Data Sources

Three datasets were collected and joined at the US county level using FIPS codes:

| # | Dataset | Scope |
|---|---------|-------|
| 1 | Voting records — US Presidential Elections | 2000–2024, county level |
| 2 | Socioeconomic data — US Counties | Year 2020 |
| 3 | Demographic data — US Counties | Year 2020 |

---

## 2. Data Cleaning & Preprocessing

Steps applied before analysis:

1. **Dropped margin-of-error columns** from socioeconomic and demographic datasets.
2. **Cleaned column names** for consistency across all three tables.
3. **Joined all three tables** on FIPS code (county identifier).
4. **Filtered to 2020** for label alignment with socioeconomic and demographic snapshots.

**Record counts after join:**

| Stage | FIPS Count |
|-------|-----------|
| Raw FIPS codes available | 3,115 |
| After joining all three tables | 2,305 |

---

## 3. Feature Engineering — Label Creation

Three prediction targets were derived from voting records:

| Label | Type | Description |
|-------|------|-------------|
| `class_label` | Binary Classification | Winner: `1` = Democrat, `0` = Republican |
| `dem_reg_votes` | Regression | Democrat votes / Total votes |
| `rep_reg_votes` | Regression | Republican votes / Total votes |

**Planned:** FIPS codes to be one-hot encoded for use in prediction. Post-model weight inspection of FIPS features is proposed to measure geographic impact.

---

## 4. Exploratory Analysis

### 4.1 Correlation Analysis — Classification Label (`class_label`)

Pearson correlation of all numerical features against the binary winner label.

Education, race, and age were the strongest signals for all education. 

Advanced degree attainment (Master's, Professional, Doctorate, Bachelor's) are the top positive correlates with Democratic outcomes across all three labels. Conversely, high school diploma, GED, and some college without a degree are the top positive correlates for Republican vote share.

Asian and Black/African American population counts positively correlated with Democratic outcomes. White-alone population positively correlated with Republican outcomes.

Older generation are positively correlated with Republican outcomes.

There are general symmetry of positive and negative correlation on these most/least correlated features across all labels.

#### Top Positive Correlates (Democrat Winner)

| Feature | Correlation |
|---------|------------|
| Total Master's degree | 0.499 |
| Total Professional school degree | 0.478 |
| Total Asian alone | 0.416 |
| Total Doctorate degree | 0.412 |
| Total Bachelor's degree | 0.393 |
| Total Female 25 to 29 years | 0.292 |
| Total Black or African American alone | 0.288 |
| Total Income ≥ poverty level — Female 25 to 34 years | 0.276 |
| Total In labor force — Civilian — Unemployed | 0.271 |
| Total Female 30 to 34 years | 0.262 |
| Total Two or more races | 0.259 |
| Median household income (2020 inflation-adjusted) | 0.236 |
| Total Income ≥ poverty level — Male 25 to 34 years | 0.232 |
| Total Income < poverty level — Male 18 to 24 years | 0.215 |
| Total Female 22 to 24 years | 0.210 |

#### Top Negative Correlates (Republican Winner)

| Feature | Correlation |
|---------|------------|
| Total White alone | -0.457 |
| Total Regular high school diploma | -0.399 |
| Total Some college, less than 1 year | -0.304 |
| Total GED or alternative credential | -0.232 |
| Total Income ≥ poverty level — Male 75 years and over | -0.231 |
| Total Income ≥ poverty level — Female 75 years and over | -0.218 |
| Total Female 75 to 79 years | -0.210 |
| Total Male 80 to 84 years | -0.225 |
| Total Male 75 to 79 years | -0.201 |
| Total Female 80 to 84 years | -0.196 |
| Total Income ≥ poverty level — Male 55 to 64 years | -0.195 |
| Total Income ≥ poverty level — Male 65 to 74 years | -0.193 |
| Total Not in labor force | -0.173 |
| Total Income ≥ poverty level — Male | -0.170 |
| Total Male 70 to 74 years | -0.168 |

---

### 4.2 Correlation Analysis — Democrat Vote Share

#### Top Positive Correlates

| Feature | Correlation |
|---------|------------|
| Total Master's degree | 0.588 |
| Total Professional school degree | 0.562 |
| Total Doctorate degree | 0.453 |
| Total Asian alone | 0.440 |
| Total Bachelor's degree | 0.430 |
| Total Black or African American alone | 0.367 |
| Total In labor force — Civilian — Unemployed | 0.339 |
| Total Female 25 to 29 years | 0.333 |
| Total Income ≥ poverty level — Female 25 to 34 years | 0.298 |
| Total Female 30 to 34 years | 0.297 |
| Total Two or more races | 0.276 |
| Median household income (2020 inflation-adjusted) | 0.275 |
| Total Income ≥ poverty level — Male 25 to 34 years | 0.270 |
| Total Female 22 to 24 years | 0.252 |
| Total Two or more races (excl. some other + 3+ races) | 0.247 |

#### Top Negative Correlates

| Feature | Correlation |
|---------|------------|
| Total White alone | -0.546 |
| Total Regular high school diploma | -0.438 |
| Total Some college, less than 1 year | -0.301 |
| Total Income ≥ poverty level — Male 75 years and over | -0.275 |
| Total Male 80 to 84 years | -0.272 |
| Total GED or alternative credential | -0.249 |
| Total Female 80 to 84 years | -0.248 |
| Total Income ≥ poverty level — Female 75 years and over | -0.240 |
| Total Female 75 to 79 years | -0.244 |
| Total Male 75 to 79 years | -0.223 |
| Total 8th grade | -0.223 |
| Total Income ≥ poverty level — Male 55 to 64 years | -0.218 |
| Total Income ≥ poverty level — Male | -0.211 |
| Total Income ≥ poverty level — Male 65 to 74 years | -0.195 |
| Total Female 85 years and over | -0.187 |

---

### 4.3 Correlation Analysis — Republican Vote Share

#### Top Positive Correlates

| Feature | Correlation |
|---------|------------|
| Total Regular high school diploma | 0.448 |
| Total Some college, less than 1 year | 0.292 |
| Total Male 80 to 84 years | 0.276 |
| Total Income ≥ poverty level — Male 75 years and over | 0.275 |
| Total Female 80 to 84 years | 0.256 |
| Total GED or alternative credential | 0.256 |
| Total Female 75 to 79 years | 0.251 |
| Total Income ≥ poverty level — Female 75 years and over | 0.247 |
| Total 8th grade | 0.230 |
| Total Male 75 to 79 years | 0.224 |
| Total Income ≥ poverty level — Male 55 to 64 years | 0.211 |
| Total Income ≥ poverty level — Male | 0.196 |
| Total Income ≥ poverty level — Male 65 to 74 years | 0.194 |
| Total Income < poverty level — Female 75 years and over | 0.191 |
| Total Female 85 years and over | 0.190 |

#### Top Negative Correlates

| Feature | Correlation |
|---------|------------|
| Total Master's degree | -0.595 |
| Total Professional school degree | -0.566 |
| Total Doctorate degree | -0.461 |
| Total Bachelor's degree | -0.442 |
| Total Asian alone | -0.441 |
| Total Black or African American alone | -0.348 |
| Total In labor force — Civilian — Unemployed | -0.332 |
| Total Female 25 to 29 years | -0.332 |
| Median household income (2020 inflation-adjusted) | -0.286 |
| Total Income ≥ poverty level — Female 25 to 34 years | -0.304 |
| Total Female 30 to 34 years | -0.299 |
| Total Income ≥ poverty level — Male 25 to 34 years | -0.278 |
| Total Two or more races | -0.277 |
| Total Female 22 to 24 years | -0.257 |
| Total Two or more races Two races excluding Some other race, and three or more races | -0.255104

---

### 4.4 Standard Deviation Analysis

Standard deviation computed on all numerical features (normalized) to assess county-to-county spread.

Income levels and demographic distribution had the highest variance, indicating geographic impact of the dataset. County encoding could be a strong indicator on prediction.

Younger population and many of the poverty level dataset are not very varied, indicating weak signals for prediction.

#### Highest Variance Features

| Feature | Std Dev |
|---------|---------|
| Median household income (2020 inflation-adjusted) | 0.2592 |
| Total White alone | 0.1538 |
| Total Black or African American alone | 0.1248 |
| Total In labor force — Civilian — Employed | 0.0803 |
| Total Not in labor force | 0.0777 |
| Total In labor force | 0.0777 |
| Total In labor force — Civilian | 0.0773 |
| Total American Indian and Alaska Native alone | 0.0719 |
| Total Regular high school diploma | 0.0667 |
| Total Income < poverty level | 0.0600 |
| Total Income ≥ poverty level | 0.0600 |
| Total Bachelor's degree | 0.0575 |
| Total Some other race alone | 0.0420 |
| Total Income ≥ poverty level — Male | 0.0351 |
| Total Income < poverty level — Female | 0.0338 |

#### Lowest Variance Features

| Feature | Std Dev |
|---------|---------|
| Total Income ≥ poverty level — Male 15 years | 0.00279 |
| Total 4th grade | 0.00273 |
| Total Income ≥ poverty level — Female 5 years | 0.00272 |
| Total Income ≥ poverty level — Female 15 years | 0.00268 |
| Total 2nd grade | 0.00257 |
| Total Income < poverty level — Male 75 years and over | 0.00247 |
| Total Income < poverty level — Female 16 and 17 years | 0.00227 |
| Total Income < poverty level — Male 16 and 17 years | 0.00209 |
| Total Income < poverty level — Male 15 years | 0.00166 |
| Total 1st grade | 0.00160 |
| Total Income < poverty level — Female 5 years | 0.00154 |
| Total Income < poverty level — Male 5 years | 0.00149 |
| Total Income < poverty level — Female 15 years | 0.00145 |
| Total Kindergarten | 0.00074 |
| Total Nursery school | 0.00049 |
