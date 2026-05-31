# Sid — Data Preprocessing

**Notebook:** `sid_preprocessing.ipynb`  
**Branch:** `siddhesh`  
**Output:** `../data/processed/merged_county_dataset.csv`

---

## Overview

Merges 7 ACS Census tables with MIT Election Lab county-level presidential results into a single clean dataset for modeling.

---

## Data Sources

| Source | Description |
|--------|-------------|
| MIT Election Lab `countypres_2000-2024.csv` | County-level presidential vote counts |
| ACS B01001 | Age and sex by county |
| ACS B02001 | Racial composition |
| ACS B15003 | Educational attainment |
| ACS B19013 | Median household income |
| ACS B17001 | Poverty status |
| ACS B23025 | Employment and labor force |
| ACS B25001 | Housing units |

---

## Pipeline Steps

### 1. Load Raw Data
All 8 CSV files loaded into separate dataframes.

### 2. ACS Cleaning Function
A shared `clean_acs()` function applied to all 7 ACS tables:
- Drops row 0 (ACS description row, not real data)
- Extracts `county_fips` from last 5 chars of `GEO_ID`
- Keeps only estimate columns (ending in `E`), excludes `NAME` and `GEO_ID`
- Converts to numeric with `errors='coerce'`
- Replaces ACS sentinel value `-666666666` with `NaN`
- Zero-pads FIPS to 5 digits with `str.zfill(5)`

### 3. Bug Fix — ACS Sentinel Value
The Census Bureau uses `-666666666` for suppressed/unavailable data. Without replacing this, all statistical computations on affected columns produce incorrect results.

### 4. Bug Fix — MIT Election Mode Filter
The MIT dataset reports votes by mode: `TOTAL`, `ABSENTEE`, `EARLY VOTING`, `PROVISIONAL` etc.  
Filtering only `TOTAL` silently drops 849 counties that report exclusively by mode.

**Fix:** Use `TOTAL` where available, aggregate all modes for counties without a `TOTAL` row.

| Approach | Counties |
|----------|----------|
| TOTAL only (reference) | 2,305 |
| Fixed (TOTAL + mode fallback) | 3,154 |

### 5. Election Data Preparation
- Filter to 2020 only
- Keep Democrat and Republican only
- Pivot to one row per county with `democrat_votes` and `republican_votes` columns
- Create targets: `party_winner` (0/1) and `dem_vote_share` (float)

### 6. FIPS Overlap Check

| Category | Count |
|----------|-------|
| ACS counties | 3,221 |
| MIT counties | 3,154 |
| Common (merged) | 3,115 |
| ACS only dropped | 106 (mostly Alaska) |
| MIT only dropped | 39 |

### 7. Final Merge
Inner join ACS merged table with election pivot on `county_fips`.

### 8. Save
Output saved to `../data/processed/merged_county_dataset.csv`

---

## Final Dataset

| Metric | Value |
|--------|-------|
| Shape | 3,115 × 162 |
| Missing values | 1 (median income, 1 county) |
| Republican-winning counties | 2,576 (82.7%) |
| Democrat-winning counties | 539 (17.3%) |
| States covered | 51 (including DC) |
