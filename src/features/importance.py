import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

Xurban = pd.read_csv(PROJECT_ROOT / "data/clean_urban_air/2_X_test.csv")

Xfield = pd.read_csv(PROJECT_ROOT / "data/resilient_fields/3_X_train.csv")
yurban = pd.read_csv(PROJECT_ROOT / "data/clean_urban_air/2_y_test.csv")

urban = pd.read_csv(PROJECT_ROOT / "data/savanna_preserve/1_y_test.csv")
yfield = pd.read_csv(PROJECT_ROOT / "data/resilient_fields/3_y_train.csv")

print("Urban Dataset Info:")
print(f"Features shape: {Xurban.shape}")
print(f"Target shape: {yurban.shape}")
print(f"Features columns: {list(Xurban.columns)}")
print(f"Target columns: {list(yurban.columns)}")

# Prepare the data
# First, let's examine the data types
print(f"\nXurban data types:")
print(Xurban.dtypes)
print(f"\nurban data types:")
print(urban.dtypes)

# Remove location_id and handle datetime columns
X = Xurban.copy()
X = X.drop(['Unnamed: 0'], axis=1)
# Remove location_id if it exists
if 'location_id' in X.columns:
    X = X.drop(['location_id'], axis=1)

# Handle datetime columns
datetime_columns = []
for col in X.columns:
    if X[col].dtype == 'object':
        # Try to convert to datetime
        try:
            pd.to_datetime(X[col])
            datetime_columns.append(col)
            print(f"Found datetime column: {col}")
        except:
            # If it's not datetime, check if it's categorical
            if X[col].nunique() < 20:  # Assume categorical if less than 20 unique values
                print(f"Found categorical column: {col}, unique values: {X[col].unique()}")
                # Convert categorical to numeric
                X[col] = pd.Categorical(X[col]).codes
            else:
                print(f"Warning: Could not handle column {col}, dropping it")
                X = X.drop([col], axis=1)

# Remove datetime columns
if datetime_columns:
    print(f"Removing datetime columns: {datetime_columns}")
    X = X.drop(datetime_columns, axis=1)

# Convert remaining object columns to numeric if possible
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            print(f"Warning: Could not convert {col} to numeric, dropping it")
            X = X.drop([col], axis=1)

# Remove any rows with NaN values
initial_shape = X.shape[0]
X = X.dropna()
if X.shape[0] != initial_shape:
    print(f"Removed {initial_shape - X.shape[0]} rows with NaN values")

# Get target variable
if 'us_aqi' in urban.columns:
    y = urban['us_aqi']
else:
    y = urban.iloc[:, 0]  # Take first column if temperature_2m doesn't exist

# Align target with features (in case we dropped rows)
y = y.iloc[X.index]

print(f"\nFinal data shapes after cleaning:")
print(f"X (features): {X.shape}")
print(f"y (target): {y.shape}")
print(f"Feature columns: {list(X.columns)}")

# Check for any remaining non-numeric data
print(f"\nFinal data types:")
print(X.dtypes)

# Ensure all features are numeric
if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
    print("Warning: Some columns are still not numeric!")
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            print(f"Non-numeric column: {col} (type: {X[col].dtype})")
            X = X.drop([col], axis=1)

# =============================================================================
# METHOD 1: Random Forest Feature Importance
# =============================================================================

print("\n" + "="*60)
print("RANDOM FOREST FEATURE IMPORTANCE - urban DATA")
print("="*60)

# Check if we have any features left
if X.shape[1] == 0:
    print("Error: No valid numeric features found!")
    exit()

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X, y)

# Get feature importance
rf_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 15 Most Important Features (Random Forest):")
print("-" * 50)
for i, (feature, importance) in enumerate(rf_importance.head(15).items()):
    print(f"{i+1:2d}. {feature:<40} {importance:.6f}")

# =============================================================================
# METHOD 2: Linear Regression Coefficients
# =============================================================================

print("\n" + "="*60)
print("LINEAR REGRESSION COEFFICIENTS - urban DATA")
print("="*60)

# Standardize features for fair coefficient comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_scaled_df, y)

# Get feature importance as absolute coefficients
lr_importance = pd.Series(
    np.abs(lr.coef_),
    index=X.columns
).sort_values(ascending=False)

print("\nTop 15 Most Important Features (Linear Regression - Absolute Coefficients):")
print("-" * 70)
for i, (feature, importance) in enumerate(lr_importance.head(15).items()):
    print(f"{i+1:2d}. {feature:<40} {importance:.6f}")
