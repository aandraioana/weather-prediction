"""Anomaly detection using Isolation Forest and PCA visualization."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data(folder):
    """Load training data for anomaly detection."""
    train_df = pd.read_csv(PROJECT_ROOT / f"data/{folder}/train.csv")

    # Parse date and extract hour for coloring
    train_df["date"] = pd.to_datetime(train_df["date"])
    train_df["hour"] = train_df["date"].dt.hour

    return train_df


def detect_anomalies(name, folder, feature_prefix):
    """Detect and visualize anomalies in a dataset."""
    print(f"\n{'='*60}")
    print(f"ANOMALY DETECTION: {name}")
    print('='*60)

    df = load_data(folder)

    # Select numeric columns with the prefix
    numeric_cols = [c for c in df.columns if feature_prefix in c.lower() and df[c].dtype in [np.float64, np.int64]]

    if not numeric_cols:
        print(f"No columns found with prefix '{feature_prefix}'")
        return

    print(f"Analyzing {len(numeric_cols)} columns: {numeric_cols[:3]}...")

    X = df[numeric_cols].fillna(0)
    hours = df["hour"].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Print feature statistics
    print("\nFeature Statistics:")
    print(f"{'Feature':<40} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 80)
    for col in numeric_cols[:5]:
        print(f"{col:<40} {df[col].min():>10.2f} {df[col].max():>10.2f} {df[col].mean():>10.2f} {df[col].std():>10.2f}")

    # PCA for visualization
    pca = PCA(n_components=min(3, len(numeric_cols)))
    X_pca = pca.fit_transform(X_scaled)

    print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    predictions = iso.fit_predict(X_pca)

    n_anomalies = (predictions == -1).sum()
    print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.1f}%)")

    # Plot
    if X_pca.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        normal = predictions == 1
        anomaly = predictions == -1

        ax.scatter(X_pca[normal, 0], X_pca[normal, 1], c='blue', alpha=0.5, s=30, label='Normal')
        scatter = ax.scatter(X_pca[anomaly, 0], X_pca[anomaly, 1], c=hours[anomaly],
                            cmap='viridis', alpha=0.8, s=50, edgecolor='k', label='Anomaly')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Anomaly Detection: {name}\n(colored by hour)')
        ax.legend()
        ax.grid(alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Hour')

        out = PROJECT_ROOT / f"anomalies_{name.replace(' ', '_')}.png"
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"Saved: {out}")


def main():
    # Resilient Fields - precipitation features
    detect_anomalies(
        "Resilient Precipitation",
        "resilient_fields",
        "precipitation"
    )

    # Resilient Fields - irradiance features
    detect_anomalies(
        "Resilient Irradiance",
        "resilient_fields",
        "irradiance"
    )

    # Savanna - temperature features
    detect_anomalies(
        "Savanna Temperature",
        "savanna_preserve",
        "temperature"
    )

    # Savanna - humidity features
    detect_anomalies(
        "Savanna Humidity",
        "savanna_preserve",
        "humidity"
    )


if __name__ == "__main__":
    main()
