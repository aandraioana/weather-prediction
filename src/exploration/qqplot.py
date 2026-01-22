"""Q-Q plots for comparing distributions of weather features."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data(folder):
    """Load training data."""
    train_df = pd.read_csv(PROJECT_ROOT / f"data/{folder}/train.csv")
    train_df["date"] = pd.to_datetime(train_df["date"])
    return train_df


def plot_qqplots(name, folder, feature1_prefix, feature2_prefix):
    """Create Q-Q plots comparing two feature types across days."""
    print(f"\n{'='*60}")
    print(f"Q-Q PLOTS: {name}")
    print('='*60)

    df = load_data(folder)

    # Find columns with both prefixes
    cols1 = sorted([c for c in df.columns if f"{feature1_prefix}_previous_day" in c],
                   key=lambda x: int(x.split("day")[-1]))
    cols2 = sorted([c for c in df.columns if f"{feature2_prefix}_previous_day" in c],
                   key=lambda x: int(x.split("day")[-1]))

    n_days = min(len(cols1), len(cols2), 7)

    if n_days == 0:
        print(f"No matching columns found for {feature1_prefix} and {feature2_prefix}")
        return

    print(f"Comparing {n_days} days of {feature1_prefix} vs {feature2_prefix}")

    fig, axes = plt.subplots(2, n_days, figsize=(4*n_days, 8))

    for day in range(n_days):
        col1 = cols1[day]
        col2 = cols2[day]

        data1 = df[col1].dropna()
        data2 = df[col2].dropna()

        # Histogram subplot
        ax_hist = axes[0, day]
        sns.histplot(data1, color='blue', alpha=0.5, label=feature1_prefix, kde=True, ax=ax_hist)
        sns.histplot(data2, color='red', alpha=0.5, label=feature2_prefix, kde=True, ax=ax_hist)
        ax_hist.set_title(f'Day {day+1} Distribution')
        ax_hist.legend(fontsize=8)
        ax_hist.set_xlabel('')

        # Q-Q plot subplot
        ax_qq = axes[1, day]

        # Remove outliers for cleaner Q-Q plot
        q1_1, q3_1 = np.percentile(data1, [25, 75])
        iqr_1 = q3_1 - q1_1
        mask1 = (data1 >= q1_1 - 1.5*iqr_1) & (data1 <= q3_1 + 1.5*iqr_1)

        q1_2, q3_2 = np.percentile(data2, [25, 75])
        iqr_2 = q3_2 - q1_2
        mask2 = (data2 >= q1_2 - 1.5*iqr_2) & (data2 <= q3_2 + 1.5*iqr_2)

        clean1 = data1[mask1].values
        clean2 = data2[mask2].values

        # Align lengths for Q-Q plot
        min_len = min(len(clean1), len(clean2))
        if min_len > 10:
            sm.qqplot_2samples(clean1[:min_len], clean2[:min_len], ax=ax_qq)
            ax_qq.set_title(f'Day {day+1} Q-Q')
        else:
            ax_qq.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax_qq.set_title(f'Day {day+1} Q-Q')

    plt.suptitle(f'{name}: {feature1_prefix} vs {feature2_prefix}', fontsize=14)
    plt.tight_layout()

    out = PROJECT_ROOT / f"qqplot_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved: {out}")


def main():
    # Savanna: Temperature vs Humidity
    plot_qqplots(
        "Savanna",
        "savanna_preserve",
        "temperature_2m",
        "relative_humidity_2m"
    )

    # Resilient Fields: Precipitation vs Irradiance
    plot_qqplots(
        "Resilient Fields",
        "resilient_fields",
        "precipitation",
        "global_tilted_irradiance"
    )


if __name__ == "__main__":
    main()
