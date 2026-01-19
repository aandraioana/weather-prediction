import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Read the files
print("Reading files...")
train_features = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_X_train.csv')
train_labels = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_y_train.csv')
test_features = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_X_test.csv')
test_labels = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_y_test.csv')

# Convert date columns to datetime for proper merging
train_features['date'] = pd.to_datetime(train_features['date'])
train_labels['date'] = pd.to_datetime(train_labels['date'])
test_features['date'] = pd.to_datetime(test_features['date'])
test_labels['date'] = pd.to_datetime(test_labels['date'])

# Merge features with labels
train_data = pd.merge(train_features, train_labels, on=['date', 'location_id'], how='inner')
test_data = pd.merge(test_features, test_labels, on=['date', 'location_id'], how='inner')
train_data.drop('Unnamed: 0_x', axis=1, inplace=True)
test_data.drop('Unnamed: 0_x', axis=1, inplace=True)
if 'season' in train_data.columns:
    season_mapping = {'winter': 1, 'spring': 2, 'summer': 3, 'fall': 4, 'autumn': 4}
    train_data['season'] = train_data['season'].str.lower().map(season_mapping)
    test_data['season'] = test_data['season'].str.lower().map(season_mapping)
# Select features (excluding non-numeric and target columns)
feature_columns = [col for col in train_data.columns if col not in ['Unnamed: 0', 'date', 'location_id', 'temperature_2m', 'relative_humidity_2m']]

# Prepare data
X_train = train_data[feature_columns].fillna(0)
y_train = train_data['temperature_2m']  # Predicting temperature
X_test = test_data[feature_columns].fillna(0)
y_test = test_data['temperature_2m']

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Accuracy (within 1°C): {np.mean(np.abs(y_test - y_pred) <= 1) * 100:.1f}%")
print(f"Accuracy (within 2°C): {np.mean(np.abs(y_test - y_pred) <= 2) * 100:.1f}%")
