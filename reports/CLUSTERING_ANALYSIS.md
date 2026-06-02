# Clustering Analysis

## Objective

The objective of the clustering analysis was to identify natural groups of U.S. counties based on demographic, socioeconomic, and political characteristics. Unlike the supervised learning models used earlier in the project, clustering was performed without using county labels as prediction targets. The goal was to discover whether counties naturally form meaningful groups that share similar demographic and voting patterns.

---

# Experiment 1: K-Means Clustering Using the Full Feature Set

The first clustering experiment used the complete feature set available in the dataset. This included demographic variables, socioeconomic indicators, engineered features, and historical election variables.

Before clustering, all numerical features were standardized using StandardScaler to ensure that variables with larger numerical ranges would not dominate the clustering process.

The Elbow Method was applied to determine an appropriate number of clusters. Several values of K were tested, and six clusters were initially selected.

However, the resulting clusters were highly imbalanced. One cluster contained the overwhelming majority of counties while several other clusters contained only a small number of observations. The PCA visualization also showed poor separation between groups.

This indicated that clustering was being dominated by a small number of highly influential variables and that the full feature set was not producing meaningful county segments.

---

# Experiment 2: Reduced Feature Selection

To improve interpretability and cluster quality, a reduced feature set was created consisting of key demographic and socioeconomic indicators:

* Higher education rate
* Poverty rate
* White population percentage
* Black population percentage
* Asian population percentage

These variables were selected because they are widely recognized as important factors associated with political behavior and county-level demographic differences.

The goal of this feature reduction step was to remove noise introduced by hundreds of highly correlated variables and to allow clustering algorithms to focus on the most meaningful county characteristics.

---

# Experiment 3: K-Means Clustering Using the Reduced Feature Set

K-Means clustering was then repeated using the reduced feature set.

The Elbow Method was applied again and suggested that approximately five clusters provided a reasonable balance between simplicity and within-cluster variation.

The resulting clusters were substantially more balanced than those obtained using the full feature set. PCA visualization showed improved separation between groups, and the resulting cluster profiles revealed several meaningful county categories:

* Predominantly White Republican counties
* Economically disadvantaged Republican counties
* Competitive or swing counties
* Counties with large Black populations
* Highly educated and diverse urban counties

These results demonstrated that demographic and socioeconomic variables alone were capable of producing meaningful county groupings.

---

# Experiment 4: Hierarchical Clustering

To validate the findings obtained through K-Means clustering, Hierarchical Clustering was applied using the same reduced feature set.

A dendrogram was generated to visualize county relationships and identify natural cluster boundaries. Based on the dendrogram structure, five clusters were selected for analysis.

The hierarchical clustering results closely matched the K-Means findings. Similar county profiles emerged, including:

* Rural Republican counties
* Working-class Republican counties
* Competitive counties
* Black-majority Democratic counties
* Highly educated urban counties

The consistency between K-Means and Hierarchical Clustering provided strong evidence that these demographic groupings represent genuine structures within the data rather than artifacts of a specific clustering algorithm.

---

# Key Findings

Several important findings emerged from the clustering analysis:

1. County-level demographic characteristics naturally form distinct groups across the United States.

2. Education level, poverty rate, and racial composition were among the strongest factors separating counties into different clusters.

3. Counties with higher education levels and greater diversity tended to exhibit stronger Democratic voting patterns.

4. Counties with predominantly White populations and lower educational attainment tended to exhibit stronger Republican voting patterns.

5. Hierarchical Clustering and K-Means produced highly consistent county groupings, increasing confidence in the validity of the discovered clusters.

6. Feature selection significantly improved clustering quality and interpretability by reducing the influence of noisy or redundant variables.

---

# Conclusion

The clustering analysis successfully identified meaningful county segments based on demographic and socioeconomic characteristics. Initial experiments using the complete feature set resulted in highly imbalanced clusters, motivating a feature-selection step that focused on a small set of interpretable variables. Using this reduced feature set, both K-Means and Hierarchical Clustering produced clear and consistent county groupings.

These findings provide additional insight into how demographic characteristics relate to political behavior and complement the supervised classification and regression models developed elsewhere in the project.
