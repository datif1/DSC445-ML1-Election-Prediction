"""
Regression models for county-level Democratic vote share prediction.

This script trains multiple regression models to predict Democratic vote share
for each county using the processed model-ready dataset.

Target:
    dem_vote_share

Input:
    data/processed/model_ready_dataset.csv

Outputs:
    reports/regression_model_results.csv
    reports/regression_feature_importance.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "dem_vote_share"

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


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and regression target y."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    feature_columns = [
        col for col in df.columns
        if col not in NON_FEATURE_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def build_models() -> dict[str, Pipeline]:
    """Create regression model pipelines."""
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        ),
        "Decision Tree Regressor": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting Regressor": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Support Vector Regressor": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, epsilon=0.05)),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a trained regression model."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": mse ** 0.5,
        "r2_score": r2_score(y_test, y_pred),
    }


def extract_feature_importance(
    model_name: str,
    model: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature importance for tree-based and linear models."""
    trained_model = model.named_steps["model"]

    if hasattr(trained_model, "feature_importances_"):
        importance_values = trained_model.feature_importances_
    elif hasattr(trained_model, "coef_"):
        importance_values = abs(trained_model.coef_)
    else:
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    return importance_df.sort_values("importance", ascending=False)


def main() -> None:
    """Run the full regression training and evaluation workflow."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y = prepare_features(df)

    print(f"Loaded dataset: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print("Target summary:")
    print(y.describe())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    models = build_models()

    results = []
    feature_importance_frames = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = model_name
        results.append(metrics)

        importance_df = extract_feature_importance(
            model_name=model_name,
            model=model,
            feature_names=list(X.columns),
        )

        if not importance_df.empty:
            feature_importance_frames.append(importance_df)

        print(f"{model_name} results:")
        for metric_name, metric_value in metrics.items():
            if metric_name != "model":
                print(f"  {metric_name}: {metric_value:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["model", "mae", "mse", "rmse", "r2_score"]
    ].sort_values("r2_score", ascending=False)

    results_path = REPORTS_DIR / "regression_model_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nFinal regression results:")
    print(results_df)
    print(f"\nSaved regression results to: {results_path}")

    if feature_importance_frames:
        feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True)
        feature_importance_path = REPORTS_DIR / "regression_feature_importance.csv"
        feature_importance_df.to_csv(feature_importance_path, index=False)
        print(f"Saved feature importance results to: {feature_importance_path}")


if __name__ == "__main__":
    main()