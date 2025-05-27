import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
pd.set_option('display.width', 0)
urban = pd.read_csv("2 - Clean Urban Air/2_X_test.csv")


field = pd.read_csv("3 - Resilient Fields/3_X_train.csv")
print(field.head())
X = pd.read_csv("3 - Resilient Fields/3_X_train.csv")
X['date'] = pd.to_datetime(X['date'])
X['hour'] = X['date'].dt.hour
y = pd.read_csv("3 - Resilient Fields/3_y_train.csv")
#outlier detection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
print("Data dimensions:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
# Print the minimum, maximum, mean and standard deviation of each feature
print("\nFeature statistics:")
print("Feature | Min      | Max      | Mean     | Std")
print("-" * 50)
numeric_X = X.select_dtypes(include=[np.number])
hour_for_coloring = numeric_X['hour']
numeric_X = numeric_X.drop(columns=["Unnamed: 0", "location_id", "date","month","season", "average_precipitation",'hour'], errors="ignore")
precipitation_columns = [col for col in numeric_X.columns if "precipitation" in col.lower()]
num_plots = len(precipitation_columns)
# Calculate rows and columns for a nearly square layout
num_cols = 3  # You can adjust this number based on preference
num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division

# Create a figure and a set of subplots
# figsize is adjusted to accommodate multiple plots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
# Flatten the axes array if there's more than one row/column, for easier iteration
axes = axes.flatten()

for i, col in enumerate(precipitation_columns):
    ax = axes[i]  # Get the current subplot axis

    # Create the histogram for each column on its own subplot
    ax.hist(numeric_X[col], bins=20, edgecolor='black', alpha=0.7)

    # Add titles and labels for the current subplot
    ax.set_title(f'Distribution of {col}', fontsize=12)
    ax.set_xlabel('Precipitation Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.75, linestyle='--')

    # Customize ticks
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

# Hide any unused subplots if the number of plots is less than num_rows * num_cols
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title for the entire figure
fig.suptitle('Histograms of Precipitation Features', fontsize=20, y=1.02)

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust rect to make space for suptitle

# Display the plot
plt.show()
irradiance_columns = [col for col in numeric_X.columns if "irradiance" in col.lower()]
num_plots = len(irradiance_columns)
# Calculate rows and columns for a nearly square layout
num_cols = 3  # You can adjust this number based on preference
num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division

# Create a figure and a set of subplots
# figsize is adjusted to accommodate multiple plots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
# Flatten the axes array if there's more than one row/column, for easier iteration
axes = axes.flatten()

for i, col in enumerate(irradiance_columns):
    ax = axes[i]  # Get the current subplot axis

    # Create the histogram for each column on its own subplot
    ax.hist(numeric_X[col], bins=20, edgecolor='black', alpha=0.7)

    # Add titles and labels for the current subplot
    ax.set_title(f'Distribution of {col}', fontsize=12)
    ax.set_xlabel('irradiance Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.75, linestyle='--')

    # Customize ticks
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

# Hide any unused subplots if the number of plots is less than num_rows * num_cols
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Add a main title for the entire figure
fig.suptitle('Histograms of irradiance Features', fontsize=20, y=1.02)

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust rect to make space for suptitle

# Display the plot
plt.show()
for i, col in enumerate(numeric_X.columns):
    print(f"{col:10} | {numeric_X[col].min():8.3f} | {numeric_X[col].max():8.3f} | {numeric_X[col].mean():8.3f} | {numeric_X[col].std():8.3f}")
feature_stds = np.std(numeric_X, axis=0)
max_std_idx = np.argmax(feature_stds)
print(f"Feature with largest standard deviation: F{max_std_idx} (std = {feature_stds[max_std_idx]:.3f})")
print("This could indicate that outliers are characterized by this feature.")
print("\n=== PRE-PROCESSING (Z-SCORE STANDARDIZATION) ===")
# Manual Z-score standardization
X_s = (numeric_X - np.mean(numeric_X, axis=0)) / np.std(numeric_X, axis=0)

# Verify standardization
print("Verification of manual standardization:")
print(f"Mean of standardized data (should be ~0): {np.mean(X_s, axis=0)}")
print(f"Std of standardized data (should be ~1): {np.std(X_s, axis=0)}")




#github code
import seaborn as sns
def plot_correlation_matrix(numerical_data):
  scaler = StandardScaler()
  std_x = scaler.fit_transform(numerical_data)
  df_x = pd.DataFrame(data=std_x)
  plt.figure(figsize=(16,14))
  return sns.heatmap(df_x.cov())
plot_correlation_matrix(numeric_X)
# Use sklearn standardization for further analysis
scaler = StandardScaler()
numeric_X = scaler.fit_transform(numeric_X)

print("\n=== DATA VISUALIZATION ===")
Xpca = PCA(n_components=3).fit_transform(numeric_X)
model = IsolationForest(contamination=0.05, random_state=42)

# Fit the model to your data (X or Xpca)
# The model learns what "normal" data looks like
model.fit(Xpca) # or model.fit(numeric_X) if you want to detect outliers in original space

# Predict outliers: -1 for outliers, 1 for inliers
# or get a decision_function (lower score = more anomalous)
predictions = model.predict(Xpca)
outlier_scores = model.decision_function(Xpca) # Lower score = more anomalous

# Now use these predictions to define your normal and anomaly indices for plotting
idx_normal = predictions == 1
idx_ano = predictions == -1

# Plotting the reduced dataset based on unsupervised detection results
if Xpca.shape[1] >= 3:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    # Plot normal points in a single color (e.g., blue)
    ax.scatter(Xpca[idx_normal, 0], Xpca[idx_normal, 1], Xpca[idx_normal, 2],
              c='blue', label='Normal', alpha=0.6, s=50)

    # Plot anomalous points, colored by their month
    # We only pass the month values for the anomalous points to 'c'
    scatter_anom = ax.scatter(Xpca[idx_ano, 0], Xpca[idx_ano, 1], Xpca[idx_ano, 2],
                              c=hour_for_coloring[idx_ano], cmap='viridis',
                              label='Anomaly by hour', alpha=0.8, s=60, edgecolor='k') # Added edgecolor for better visibility

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA Visualization: Normal Points vs. Anomalies by hour')

    # Create a separate legend handle for the anomaly colorbar
    # We don't want a legend entry for each month, but a colorbar
    handles, labels = ax.get_legend_handles_labels()
    # Remove the default scatter plot legend entry for the anomalies if it's there
    # (Matplotlib might create one based on the 'label' argument)
    # A cleaner way is to ensure 'label' for scatter_anom is only for the legend key, not individual months.
    # We rely on the colorbar for month mapping.
    if 'Anomaly by hour' in labels:
        anom_idx = labels.index('Anomaly by hour')
        handles.pop(anom_idx)
        labels.pop(anom_idx)
    ax.legend(handles, labels)


    # Add a color bar specifically for the anomalous points' months
    cbar = fig.colorbar(scatter_anom, ax=ax, pad=0.1)
    cbar.set_label('hour of Anomaly')
    cbar.set_ticks(np.arange(0, 23))
    cbar.set_ticklabels([f'hour {m}' for m in range(0, 23)])

    plt.show()
else:
    print("Cannot plot in 3D: Xpca has fewer than 3 dimensions after PCA.")
    # Fallback to 2D plot if 3D is not possible
    if Xpca.shape[1] >= 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()

        # Plot normal points in a single color
        ax.scatter(Xpca[idx_normal, 0], Xpca[idx_normal, 1],
                  c='blue', label='Normal', alpha=0.6, s=50)

        # Plot anomalous points, colored by their month
        scatter_anom = ax.scatter(Xpca[idx_ano, 0], Xpca[idx_ano, 1],
                                  c=hour_for_coloring[idx_ano], cmap='viridis',
                                  label='Anomaly by hour', alpha=0.8, s=60, edgecolor='k')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA Visualization: Normal Points vs. Anomalies by hour (2D)')
        ax.legend()

        cbar = fig.colorbar(scatter_anom, ax=ax, pad=0.05)
        cbar.set_label('hour of Anomaly')
        cbar.set_ticks(np.arange(0, 23))
        cbar.set_ticklabels([f'hour {m}' for m in range(0, 23)])

        plt.show()





