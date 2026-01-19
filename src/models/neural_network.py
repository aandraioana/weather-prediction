import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Tunable hyperparameters
HIDDEN_LAYERS = [64, 32]
LEARNING_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 32
DROPOUT_RATE = 0.1

class WeatherPredictor(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate=0.2):
        super(WeatherPredictor, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_and_prepare_data():
    print("Reading files...")
    train_features = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_X_train.csv')
    train_labels = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_y_train.csv')
    test_features = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_X_test.csv')
    test_labels = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_y_test.csv')

    # Convert date columns to datetime
    train_features['date'] = pd.to_datetime(train_features['date'])
    train_labels['date'] = pd.to_datetime(train_labels['date'])
    test_features['date'] = pd.to_datetime(test_features['date'])
    test_labels['date'] = pd.to_datetime(test_labels['date'])

    # Merge features with labels
    train_data = pd.merge(train_features, train_labels, on=['date', 'location_id'], how='inner')
    test_data = pd.merge(test_features, test_labels, on=['date', 'location_id'], how='inner')

    # Drop unnecessary columns
    train_data.drop('Unnamed: 0_x', axis=1, inplace=True, errors='ignore')
    test_data.drop('Unnamed: 0_x', axis=1, inplace=True, errors='ignore')

    # Handle season mapping
    if 'season' in train_data.columns:
        season_mapping = {'winter': 1, 'spring': 2, 'summer': 3, 'fall': 4, 'autumn': 4}
        train_data['season'] = train_data['season'].str.lower().map(season_mapping)
        test_data['season'] = test_data['season'].str.lower().map(season_mapping)

    # Select features
    feature_columns = [col for col in train_data.columns
                      if col not in ['Unnamed: 0', 'Unnamed: 0_y', 'date', 'location_id',
                                    'temperature_2m', 'relative_humidity_2m']]

    X_train = train_data[feature_columns].fillna(0).values
    y_train = train_data['temperature_2m'].values
    X_test = test_data[feature_columns].fillna(0).values
    y_test = test_data['temperature_2m'].values

    return X_train, y_train, X_test, y_test, feature_columns


def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def evaluate_model(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = model(X_test_tensor).squeeze().numpy()

    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)

    print(f"\nResults:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Accuracy (within 1°C): {np.mean(np.abs(y_test_original - predictions) <= 1) * 100:.1f}%")
    print(f"Accuracy (within 2°C): {np.mean(np.abs(y_test_original - predictions) <= 2) * 100:.1f}%")

    return predictions, y_test_original


def main():
    print("=" * 60)
    print("NEURAL NETWORK - Weather Prediction")
    print("=" * 60)
    print(f"\nHyperparameters:")
    print(f"  Hidden layers: {HIDDEN_LAYERS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    print()

    X_train, y_train, X_test, y_test, feature_columns = load_and_prepare_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_columns)}")


    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()


    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    input_size = X_train_scaled.shape[1]
    model = WeatherPredictor(input_size, HIDDEN_LAYERS, DROPOUT_RATE)
    print(f"\nModel architecture:\n{model}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    evaluate_model(model, X_test_scaled, y_test_scaled, scaler_y)


if __name__ == "__main__":
    main()
