"""
Compare Linear Regression vs Neural Network (MLP) for weather prediction.
Uses sklearn for both models - lightweight, no PyTorch needed.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data(folder, target_col, drop_cols):
    """Load train and test data."""
    base = PROJECT_ROOT / f"data/{folder}"

    train_df = pd.read_csv(base / "train.csv")
    test_df = pd.read_csv(base / "test.csv")

    # Handle season encoding
    if "season" in train_df.columns:
        season_map = {"winter": 1, "spring": 2, "summer": 3, "fall": 4, "autumn": 4}
        train_df["season"] = train_df["season"].str.lower().map(season_map)
        test_df["season"] = test_df["season"].str.lower().map(season_map)

    # Get feature columns
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date", "location_id", target_col] + drop_cols
    ]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[target_col].values
    dates = test_df["date"].values if "date" in test_df.columns else None

    return X_train, y_train, X_test, y_test, dates, feature_cols


def train_linear_regression(X_train, y_train, X_test):
    """Train linear regression and return predictions."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_neural_network(X_train, y_train, X_test):
    """Train MLP neural network and return predictions."""
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
        verbose=False
    )

    model.fit(X_train_s, y_train)
    return model.predict(X_test_s)


def evaluate_model(y_true, y_pred, name):
    """Calculate evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    target_range = y_true.max() - y_true.min()
    thresh_5 = target_range * 0.05
    thresh_10 = target_range * 0.10
    acc_5 = np.mean(np.abs(y_true - y_pred) <= thresh_5) * 100
    acc_10 = np.mean(np.abs(y_true - y_pred) <= thresh_10) * 100

    return {
        "name": name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "acc_5pct": acc_5,
        "acc_10pct": acc_10
    }


def print_comparison(lr_metrics, nn_metrics, target_name):
    """Print comparison table."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {target_name}")
    print('='*60)

    print(f"\n{'Metric':<20} {'Linear Reg':>15} {'Neural Net':>15} {'Better':>10}")
    print("-" * 60)

    metrics = [
        ("MSE", "mse", False),
        ("RMSE", "rmse", False),
        ("MAE", "mae", False),
        ("R²", "r2", True),
        ("Accuracy ±5%", "acc_5pct", True),
        ("Accuracy ±10%", "acc_10pct", True),
    ]

    for display_name, key, higher_better in metrics:
        lr_val = lr_metrics[key]
        nn_val = nn_metrics[key]

        if higher_better:
            better = "NN" if nn_val > lr_val else "LR"
        else:
            better = "NN" if nn_val < lr_val else "LR"

        if key in ["r2"]:
            print(f"{display_name:<20} {lr_val:>15.4f} {nn_val:>15.4f} {better:>10}")
        elif key in ["acc_5pct", "acc_10pct"]:
            print(f"{display_name:<20} {lr_val:>14.1f}% {nn_val:>14.1f}% {better:>10}")
        else:
            print(f"{display_name:<20} {lr_val:>15.2f} {nn_val:>15.2f} {better:>10}")


def plot_comparison(y_test, lr_pred, nn_pred, target_name):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Scatter: LR
    axes[0].scatter(y_test, lr_pred, alpha=0.6, s=20)
    mn, mx = min(y_test.min(), lr_pred.min()), max(y_test.max(), lr_pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'k--')
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Linear Regression")
    axes[0].grid(alpha=0.3)

    # Scatter: NN
    axes[1].scatter(y_test, nn_pred, alpha=0.6, s=20, color='orange')
    mn, mx = min(y_test.min(), nn_pred.min()), max(y_test.max(), nn_pred.max())
    axes[1].plot([mn, mx], [mn, mx], 'k--')
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Neural Network")
    axes[1].grid(alpha=0.3)

    # Error comparison
    lr_errors = np.abs(y_test - lr_pred)
    nn_errors = np.abs(y_test - nn_pred)
    axes[2].hist(lr_errors, bins=30, alpha=0.6, label='Linear Reg')
    axes[2].hist(nn_errors, bins=30, alpha=0.6, label='Neural Net')
    axes[2].set_xlabel("Absolute Error")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Error Distribution")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle(f"Model Comparison: {target_name}")
    plt.tight_layout()

    out = PROJECT_ROOT / f"comparison_{target_name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved: {out}")


def run_comparison(name, folder, target_col, drop_cols):
    """Run full comparison for a dataset."""
    print(f"\n{'='*60}")
    print(f"Loading: {name}")
    print('='*60)

    X_train, y_train, X_test, y_test, dates, features = load_data(
        folder, target_col, drop_cols
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(features)}")

    # Train models
    print("Training Linear Regression...")
    lr_pred = train_linear_regression(X_train, y_train, X_test)

    print("Training Neural Network...")
    nn_pred = train_neural_network(X_train, y_train, X_test)

    # Evaluate
    lr_metrics = evaluate_model(y_test, lr_pred, "Linear Regression")
    nn_metrics = evaluate_model(y_test, nn_pred, "Neural Network")

    # Print results
    print_comparison(lr_metrics, nn_metrics, name)

    # Plot
    plot_comparison(y_test, lr_pred, nn_pred, name)

    return {
        "name": name,
        "lr": lr_metrics,
        "nn": nn_metrics
    }


def main():
    results = []

    results.append(run_comparison(
        "Savanna Temperature",
        "savanna_preserve",
        "temperature_2m",
        ["relative_humidity_2m"]
    ))

    results.append(run_comparison(
        "Urban AQI",
        "clean_urban_air",
        "us_aqi",
        ["relative_humidity_2m"]
    ))

    results.append(run_comparison(
        "Resilient Irradiance",
        "resilient_fields",
        "global_tilted_irradiance",
        ["precipitation"]
    ))

    results.append(run_comparison(
        "Resilient Precipitation",
        "resilient_fields",
        "precipitation",
        ["global_tilted_irradiance"]
    ))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<25} {'LR R²':>10} {'NN R²':>10} {'Winner':>10}")
    print("-" * 55)
    for r in results:
        winner = "NN" if r["nn"]["r2"] > r["lr"]["r2"] else "LR"
        print(f"{r['name']:<25} {r['lr']['r2']:>10.4f} {r['nn']['r2']:>10.4f} {winner:>10}")


if __name__ == "__main__":
    main()
