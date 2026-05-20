"""
Clustering analysis for county-level demographic and socioeconomic patterns.

This script groups counties into clusters using demographic and socioeconomic
features from the model-ready dataset, then compares the discovered clusters
against voting outcomes.

Input:
    data/processed/model_ready_dataset.csv

Outputs:
    reports/clustering_model_results.csv
    reports/clustering_cluster_summary.csv
    reports/clustering_county_assignments.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMNS = [
    "party_winner",
    "dem_vote_share",
]

NON_FEATURE_COLUMNS = [
    "GEO_ID",
    "NAME",
    "county_fips",
    "state",
    "county_name",
    "totalvotes",
    "democrat_votes",
    "republican_votes",
    "party_winner",
    "dem_vote_share",
]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the model-ready dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)

CLUSTER_FEATURES = [
    "higher_education_rate",
    "poverty_rate",
    "unemployment_rate",
    "white_pct",
    "black_pct",
    "asian_pct",
    "log_population",
    "median_income",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select engineered demographic and socioeconomic features for clustering."""
    missing_features = [col for col in CLUSTER_FEATURES if col not in df.columns]

    if missing_features:
        raise ValueError(f"Missing clustering features: {missing_features}")

    return df[CLUSTER_FEATURES].copy()


def build_preprocessing_pipeline() -> Pipeline:
    """Create preprocessing pipeline for clustering."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def evaluate_kmeans_range(X_scaled, k_values: range) -> pd.DataFrame:
    """Evaluate K-Means across a range of cluster counts."""
    results = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        results.append(
            {
                "model": "KMeans",
                "n_clusters": k,
                "silhouette_score": score,
                "inertia": model.inertia_,
            }
        )

    return pd.DataFrame(results)


def fit_final_kmeans(X_scaled, n_clusters: int) -> KMeans:
    """Fit final K-Means model."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    model.fit(X_scaled)
    return model


def fit_agglomerative(X_scaled, n_clusters: int) -> tuple[AgglomerativeClustering, float]:
    """Fit agglomerative clustering and return model with silhouette score."""
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return model, score


def create_cluster_summary(df: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    """Summarize voting and demographic patterns within each cluster."""
    summary = df.groupby(cluster_column).agg(
        counties=("county_fips", "count"),
        republican_counties=("party_winner", lambda x: (x == 0).sum()),
        democrat_counties=("party_winner", lambda x: (x == 1).sum()),
        avg_dem_vote_share=("dem_vote_share", "mean"),
        median_dem_vote_share=("dem_vote_share", "median"),
        avg_total_population=("total_population", "mean"),
        avg_median_income=("median_income", "mean"),
        avg_higher_education_rate=("higher_education_rate", "mean"),
        avg_poverty_rate=("poverty_rate", "mean"),
        avg_unemployment_rate=("unemployment_rate", "mean"),
        avg_white_pct=("white_pct", "mean"),
        avg_black_pct=("black_pct", "mean"),
        avg_asian_pct=("asian_pct", "mean"),
        avg_log_population=("log_population", "mean"),
    ).reset_index()

    summary["democrat_county_share"] = (
        summary["democrat_counties"] / summary["counties"]
    )

    return summary.sort_values("avg_dem_vote_share", ascending=False)


def add_pca_coordinates(df: pd.DataFrame, X_scaled) -> pd.DataFrame:
    """Add two PCA coordinates for optional visualization."""
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(X_scaled)

    df = df.copy()
    df["pca_1"] = coordinates[:, 0]
    df["pca_2"] = coordinates[:, 1]

    print(
        "PCA explained variance ratio:",
        [round(value, 4) for value in pca.explained_variance_ratio_],
    )

    return df


def main() -> None:
    """Run clustering workflow."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X = prepare_features(df)

    print(f"Loaded dataset: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print(f"Clustering feature matrix shape: {X.shape}")

    preprocessor = build_preprocessing_pipeline()
    X_scaled = preprocessor.fit_transform(X)

    print("\nEvaluating K-Means cluster counts...")
    kmeans_results = evaluate_kmeans_range(X_scaled, range(2, 11))

    print(kmeans_results)

    best_row = kmeans_results.sort_values("silhouette_score", ascending=False).iloc[0]
    best_k = int(best_row["n_clusters"])

    print(f"\nBest K-Means cluster count by silhouette score: {best_k}")

    final_kmeans = fit_final_kmeans(X_scaled, best_k)
    kmeans_labels = final_kmeans.labels_

    agg_model, agg_score = fit_agglomerative(X_scaled, best_k)
    agg_labels = agg_model.labels_

    agg_result = pd.DataFrame(
        [
            {
                "model": "AgglomerativeClustering",
                "n_clusters": best_k,
                "silhouette_score": agg_score,
                "inertia": None,
            }
        ]
    )

    clustering_results = pd.concat(
        [kmeans_results, agg_result],
        ignore_index=True,
    )

    results_path = REPORTS_DIR / "clustering_model_results.csv"
    clustering_results.to_csv(results_path, index=False)

    output_df = df.copy()
    output_df["kmeans_cluster"] = kmeans_labels
    output_df["agglomerative_cluster"] = agg_labels
    output_df = add_pca_coordinates(output_df, X_scaled)

    summary_df = create_cluster_summary(output_df, "kmeans_cluster")

    summary_path = REPORTS_DIR / "clustering_cluster_summary.csv"
    assignments_path = REPORTS_DIR / "clustering_county_assignments.csv"

    summary_df.to_csv(summary_path, index=False)
    output_df[
        [
            "county_fips",
            "state",
            "county_name",
            "party_winner",
            "dem_vote_share",
            "kmeans_cluster",
            "agglomerative_cluster",
            "pca_1",
            "pca_2",
        ]
    ].to_csv(assignments_path, index=False)

    print(f"\nSaved clustering model results to: {results_path}")
    print(f"Saved cluster summary to: {summary_path}")
    print(f"Saved county cluster assignments to: {assignments_path}")

    print("\nK-Means cluster summary:")
    print(summary_df)


if __name__ == "__main__":
    main()