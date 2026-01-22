"""Feature importance analysis using Random Forest and Linear Regression."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data(folder, target_col, drop_cols):
    """Load training data."""
    train_df = pd.read_csv(PROJECT_ROOT / f"data/{folder}/train.csv")

    # Handle season encoding
    if "season" in train_df.columns:
        season_map = {"winter": 1, "spring": 2, "summer": 3, "fall": 4, "autumn": 4}
        train_df["season"] = train_df["season"].str.lower().map(season_map)

    feature_cols = [
        c for c in train_df.columns
        if c not in ["date", "location_id", target_col] + drop_cols
    ]

    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col].values

    return X, y, feature_cols


def analyze_importance(name, folder, target_col, drop_cols):
    """Analyze feature importance for a dataset."""
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE: {name}")
    print('='*60)

    X, y, features = load_data(folder, target_col, drop_cols)
    print(f"Features: {len(features)} | Samples: {len(X)}")

    # Random Forest importance
    print("\n--- Random Forest Importance ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    rf.fit(X, y)

    rf_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    print("\nTop 10 Most Important Features:")
    for i, (feat, imp) in enumerate(rf_importance.head(10).items()):
        print(f"  {i+1:2d}. {feat:<45} {imp:.4f}")

    # Linear Regression coefficients
    print("\n--- Linear Regression Coefficients ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_scaled, y)

    lr_importance = pd.Series(np.abs(lr.coef_), index=features).sort_values(ascending=False)

    print("\nTop 10 by Absolute Coefficient:")
    for i, (feat, coef) in enumerate(lr_importance.head(10).items()):
        print(f"  {i+1:2d}. {feat:<45} {coef:.4f}")

    return rf_importance, lr_importance


def main():
    # Savanna - Temperature
    analyze_importance(
        "Savanna Temperature",
        "savanna_preserve",
        "temperature_2m",
        ["relative_humidity_2m"]
    )

    # Urban - AQI
    analyze_importance(
        "Urban AQI",
        "clean_urban_air",
        "us_aqi",
        ["relative_humidity_2m"]
    )

    # Resilient - Irradiance
    analyze_importance(
        "Resilient Irradiance",
        "resilient_fields",
        "global_tilted_irradiance",
        ["precipitation"]
    )

    # Resilient - Precipitation
    analyze_importance(
        "Resilient Precipitation",
        "resilient_fields",
        "precipitation",
        ["global_tilted_irradiance"]
    )


if __name__ == "__main__":
    main()
