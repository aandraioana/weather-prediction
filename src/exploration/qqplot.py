import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import pandas as pd
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

urban = pd.read_csv(PROJECT_ROOT / "data/clean_urban_air/2_X_test.csv")

savannah = pd.read_csv(PROJECT_ROOT / "data/savanna_preserve/1_X_test.csv")
field = pd.read_csv(PROJECT_ROOT / "data/resilient_fields/3_X_train.csv")
print(field.head())
X = pd.read_csv(PROJECT_ROOT / "data/savanna_preserve/1_X_train.csv")
X['date'] = pd.to_datetime(X['date'])
X['hour'] = X['date'].dt.hour
fig, axes = plt.subplots(4, 4, figsize=(20, 16))  # 4x4 grid to accommodate 7 days + 1 extra
axes = axes.flatten()

for day in range(1, 8):  # Days 1 to 7
    # Column names
    humidity_col = f'relative_humidity_2m_previous_day{day}'
    temp_col = f'temperature_2m_previous_day{day}'

    # First subplot - histograms
    ax_hist = axes[(day - 1) * 2]
    sns.histplot(X[humidity_col], color='blue', label=f'humidity Day {day}', kde=True, ax=ax_hist)
    sns.histplot(X[temp_col], color='red', label=f'temperature Day {day}', kde=True, ax=ax_hist)
    ax_hist.set_xlabel('Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'Day {day} - Distribution')
    ax_hist.legend()

    # Outlier removal for humidity
    q1_precip = np.percentile(X[humidity_col], 25)
    q3_precip = np.percentile(X[humidity_col], 75)
    iqr_precip = q3_precip - q1_precip
    lower_bound_precip = q1_precip - 1.5 * iqr_precip
    upper_bound_precip = q3_precip + 1.5 * iqr_precip

    # Outlier removal for irradiance
    q1_irrad = np.percentile(X[temp_col], 25)
    q3_irrad = np.percentile(X[temp_col], 75)
    iqr_irrad = q3_irrad - q1_irrad
    lower_bound_irrad = q1_irrad - 1.5 * iqr_irrad
    upper_bound_irrad = q3_irrad + 1.5 * iqr_irrad

    # Filter data
    data_filtered = X.loc[
        (X[humidity_col] >= lower_bound_precip) & (X[humidity_col] <= upper_bound_precip) &
        (X[temp_col] >= lower_bound_irrad) & (X[temp_col] <= upper_bound_irrad)
        ]

    # Second subplot - QQ plot
    ax_qq = axes[(day - 1) * 2 + 1]
    sm.qqplot_2samples(data_filtered[humidity_col], data_filtered[temp_col], ax=ax_qq)
    ax_qq.set_title(f'Day {day} - QQ Plot (temperature vs humidity)')

# Hide extra subplots if any
for i in range(14, 16):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()
