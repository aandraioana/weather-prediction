import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_dataset(folder, target_cols, drop_cols):
    """Load train and test data from CSV files."""
    base = PROJECT_ROOT / f"data/{folder}"

    train_df = pd.read_csv(base / "train.csv")
    test_df = pd.read_csv(base / "test.csv")

    # Handle season encoding if present
    if "season" in train_df.columns:
        season_map = {"winter": 1, "spring": 2, "summer": 3, "fall": 4, "autumn": 4}
        train_df["season"] = train_df["season"].str.lower().map(season_map)
        test_df["season"] = test_df["season"].str.lower().map(season_map)

    # Select feature columns
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date", "location_id"] + target_cols + drop_cols
    ]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_cols].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[target_cols].values

    return X_train, y_train, X_test, y_test, feature_cols


def run_experiment(name, folder, targets, drop_cols):
    """Run linear regression experiment on a dataset."""
    print("\n" + "=" * 60)
    print(f"Dataset: {name}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, features = load_dataset(folder, targets, drop_cols)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(features)}")
    print(f"Targets: {targets}")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nResults:")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.4f}")

    # Per-target metrics
    y_test = np.atleast_2d(y_test)
    y_pred = np.atleast_2d(y_pred)
    if y_test.shape[0] == 1:
        y_test = y_test.T
        y_pred = y_pred.T

    for i, t in enumerate(targets):
        y_true = y_test[:, i]
        y_p = y_pred[:, i]
        target_range = y_true.max() - y_true.min()

        thresh_5pct = target_range * 0.05
        thresh_10pct = target_range * 0.10

        acc_5 = np.mean(np.abs(y_true - y_p) <= thresh_5pct) * 100
        acc_10 = np.mean(np.abs(y_true - y_p) <= thresh_10pct) * 100

        target_r2 = r2_score(y_true, y_p)
        target_mse = mean_squared_error(y_true, y_p)

        print(f"\n{t}:")
        print(f"  R²: {target_r2:.4f} | MSE: {target_mse:.2f}")
        print(f"  Accuracy ±5%: {acc_5:.1f}% | ±10%: {acc_10:.1f}%")


def main():
    run_experiment(
        name="Savanna Preserve - Temperature",
        folder="savanna_preserve",
        targets=["temperature_2m"],
        drop_cols=["relative_humidity_2m"]
    )

    run_experiment(
        name="Clean Urban Air - AQI",
        folder="clean_urban_air",
        targets=["us_aqi"],
        drop_cols=["relative_humidity_2m"]
    )

    run_experiment(
        name="Resilient Fields - Irradiance",
        folder="resilient_fields",
        targets=["global_tilted_irradiance"],
        drop_cols=["precipitation"]
    )

    run_experiment(
        name="Resilient Fields - Precipitation",
        folder="resilient_fields",
        targets=["precipitation"],
        drop_cols=["global_tilted_irradiance"]
    )


if __name__ == "__main__":
    main()
