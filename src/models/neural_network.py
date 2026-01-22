import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent

VAL_SPLIT = 0.2

# Simplified configs using sklearn MLPRegressor
CONFIGS = {
    "savanna_temperature": {
        "hidden_layers": (64, 32),
        "learning_rate": 0.001,
        "max_iter": 500,
        "early_stopping": True,
    },
    "urban_aqi": {
        "hidden_layers": (64, 32),
        "learning_rate": 0.001,
        "max_iter": 500,
        "early_stopping": True,
    },
    "resilient_irradiance": {
        "hidden_layers": (64, 32),
        "learning_rate": 0.001,
        "max_iter": 500,
        "early_stopping": True,
    },
    "resilient_precipitation": {
        "hidden_layers": (32,),
        "learning_rate": 0.001,
        "max_iter": 1000,
        "early_stopping": True,
    },
}


# =========================
# Preprocessing
# =========================

def extract_time_features(df):
    """Extract hour and day of week from date column with cyclical encoding."""
    if "date" not in df.columns:
        return df

    df["date"] = pd.to_datetime(df["date"])

    hour = df["date"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["is_weekend"] = (dow >= 5).astype(int)

    return df


def add_cyclical_encoding(df):
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df.drop("month", axis=1, inplace=True)

    if "season" in df.columns:
        if df["season"].dtype == object:
            season_map = {"winter": 1, "spring": 2, "summer": 3, "fall": 4, "autumn": 4}
            df["season"] = df["season"].str.lower().map(season_map)
        df["season_sin"] = np.sin(2 * np.pi * df["season"] / 4)
        df["season_cos"] = np.cos(2 * np.pi * df["season"] / 4)
        df.drop("season", axis=1, inplace=True)

    return df


def add_lag_differences(df, prefix):
    lag_cols = [c for c in df.columns if f"{prefix}_previous_day" in c]
    lag_cols = sorted(lag_cols, key=lambda x: int(x.split("day")[-1]))

    if len(lag_cols) >= 2:
        df[f"{prefix}_change_1d"] = df[lag_cols[0]] - df[lag_cols[1]]
    if len(lag_cols) >= 7:
        df[f"{prefix}_change_7d"] = df[lag_cols[0]] - df[lag_cols[6]]
        df[f"{prefix}_std_7d"] = df[lag_cols].std(axis=1)

    return df


def preprocess_features(df):
    df = df.copy()
    df = extract_time_features(df)
    df = add_cyclical_encoding(df)

    for col in df.columns:
        if "_previous_day1" in col:
            prefix = col.replace("_previous_day1", "")
            df = add_lag_differences(df, prefix)

    return df


# =========================
# Dataset
# =========================

def load_dataset(folder, target_cols, drop_cols):
    base = PROJECT_ROOT / f"data/{folder}"

    train_df = pd.read_csv(base / "train.csv")
    test_df = pd.read_csv(base / "test.csv")

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    train_df = preprocess_features(train_df)
    test_df = preprocess_features(test_df)

    feature_cols = [
        c for c in train_df.columns
        if c not in ["date", "location_id"] + target_cols + drop_cols
    ]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_cols].values.ravel()
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[target_cols].values.ravel()

    return X_train, y_train, X_test, y_test, feature_cols


# =========================
# Plotting
# =========================

def plot_predictions(y_true, y_pred, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual – {name}")
    plt.grid(alpha=0.3)

    out = PROJECT_ROOT / f"pred_vs_actual_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved: {out}")


def plot_error_hist(y_true, y_pred, name):
    errors = y_true - y_pred
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=40, alpha=0.75)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution – {name}")
    plt.grid(alpha=0.3)

    out = PROJECT_ROOT / f"error_hist_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved: {out}")


# =========================
# Training
# =========================

def run_experiment(name, folder, targets, drop_cols, config_name):
    cfg = CONFIGS[config_name]

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print('='*60)

    X_train, y_train, X_test, y_test, features = load_dataset(folder, targets, drop_cols)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(features)}")

    # Scale features
    scaler_X = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # Create and train model
    model = MLPRegressor(
        hidden_layer_sizes=cfg['hidden_layers'],
        learning_rate_init=cfg['learning_rate'],
        max_iter=cfg['max_iter'],
        early_stopping=cfg['early_stopping'],
        validation_fraction=VAL_SPLIT,
        random_state=42,
        verbose=False
    )

    model.fit(X_train_s, y_train)

    # Predict
    y_pred = model.predict(X_test_s)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    target_range = y_test.max() - y_test.min()
    thresh_5pct = target_range * 0.05
    thresh_10pct = target_range * 0.10
    acc_5 = np.mean(np.abs(y_test - y_pred) <= thresh_5pct) * 100
    acc_10 = np.mean(np.abs(y_test - y_pred) <= thresh_10pct) * 100

    print(f"\nResults:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Accuracy ±5%: {acc_5:.1f}%")
    print(f"  Accuracy ±10%: {acc_10:.1f}%")
    print(f"  Iterations: {model.n_iter_}")

    plot_predictions(y_test, y_pred, name)
    plot_error_hist(y_test, y_pred, name)


def main():
    run_experiment(
        "Savanna Temperature",
        "savanna_preserve",
        ["temperature_2m"],
        ["relative_humidity_2m"],
        "savanna_temperature"
    )

    run_experiment(
        "Urban AQI",
        "clean_urban_air",
        ["us_aqi"],
        ["relative_humidity_2m"],
        "urban_aqi"
    )

    run_experiment(
        "Resilient Irradiance",
        "resilient_fields",
        ["global_tilted_irradiance"],
        ["precipitation"],
        "resilient_irradiance"
    )

    run_experiment(
        "Resilient Precipitation",
        "resilient_fields",
        ["precipitation"],
        ["global_tilted_irradiance"],
        "resilient_precipitation"
    )


if __name__ == "__main__":
    main()
