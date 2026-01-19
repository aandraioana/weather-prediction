import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Neural network hyperparameters
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


def load_data():
    """Load and prepare the dataset."""
    train_features = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_X_train.csv')
    train_labels = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_y_train.csv')
    test_features = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_X_test.csv')
    test_labels = pd.read_csv(PROJECT_ROOT / 'data/savanna_preserve/1_y_test.csv')

    train_features['date'] = pd.to_datetime(train_features['date'])
    train_labels['date'] = pd.to_datetime(train_labels['date'])
    test_features['date'] = pd.to_datetime(test_features['date'])
    test_labels['date'] = pd.to_datetime(test_labels['date'])

    train_data = pd.merge(train_features, train_labels, on=['date', 'location_id'], how='inner')
    test_data = pd.merge(test_features, test_labels, on=['date', 'location_id'], how='inner')

    train_data.drop('Unnamed: 0_x', axis=1, inplace=True, errors='ignore')
    test_data.drop('Unnamed: 0_x', axis=1, inplace=True, errors='ignore')

    if 'season' in train_data.columns:
        season_mapping = {'winter': 1, 'spring': 2, 'summer': 3, 'fall': 4, 'autumn': 4}
        train_data['season'] = train_data['season'].str.lower().map(season_mapping)
        test_data['season'] = test_data['season'].str.lower().map(season_mapping)

    feature_columns = [col for col in train_data.columns
                      if col not in ['Unnamed: 0', 'Unnamed: 0_y', 'date', 'location_id',
                                    'temperature_2m', 'relative_humidity_2m']]

    X_train = train_data[feature_columns].fillna(0).values
    y_train = train_data['temperature_2m'].values
    X_test = test_data[feature_columns].fillna(0).values
    y_test = test_data['temperature_2m'].values
    test_dates = test_data['date'].values

    return X_train, y_train, X_test, y_test, test_dates


def train_linear_regression(X_train, y_train, X_test):
    """Train linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_neural_network(X_train, y_train, X_test):
    """Train neural network model."""
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale target to [0, 1] range for better training
    y_min, y_max = y_train.min(), y_train.max()
    y_train_scaled = (y_train - y_min) / (y_max - y_min)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = WeatherPredictor(X_train_scaled.shape[1], HIDDEN_LAYERS, DROPOUT_RATE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        predictions = model(X_test_tensor).squeeze().numpy()

    # Inverse transform
    return predictions * (y_max - y_min) + y_min


def save_predictions_csv(y_test, lr_pred, nn_pred, dates):
    """Save predictions to CSV file."""
    results_df = pd.DataFrame({
        'date': dates,
        'actual': y_test,
        'linear_regression_predicted': lr_pred,
        'neural_network_predicted': nn_pred,
        'lr_error': y_test - lr_pred,
        'nn_error': y_test - nn_pred,
        'lr_abs_error': np.abs(y_test - lr_pred),
        'nn_abs_error': np.abs(y_test - nn_pred)
    })

    # Sort by date
    results_df = results_df.sort_values('date').reset_index(drop=True)

    csv_path = PROJECT_ROOT / 'predictions_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")
    return results_df


def plot_comparison(y_test, lr_pred, nn_pred):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Calculate metrics
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    nn_mse = mean_squared_error(y_test, nn_pred)
    nn_r2 = r2_score(y_test, nn_pred)

    # Plot 1: Actual vs Predicted scatter plots
    ax1 = axes[0, 0]
    ax1.scatter(y_test, lr_pred, alpha=0.6, label=f'Linear Regression (R²={lr_r2:.2f})', color='blue')
    ax1.scatter(y_test, nn_pred, alpha=0.6, label=f'Neural Network (R²={nn_r2:.2f})', color='red')
    min_val = min(y_test.min(), lr_pred.min(), nn_pred.min())
    max_val = max(y_test.max(), lr_pred.max(), nn_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Temperature (°C)')
    ax1.set_ylabel('Predicted Temperature (°C)')
    ax1.set_title('Actual vs Predicted Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prediction errors distribution
    ax2 = axes[0, 1]
    lr_errors = y_test - lr_pred
    nn_errors = y_test - nn_pred
    ax2.hist(lr_errors, bins=30, alpha=0.6, label=f'Linear Regression (MSE={lr_mse:.2f})', color='blue')
    ax2.hist(nn_errors, bins=30, alpha=0.6, label=f'Neural Network (MSE={nn_mse:.2f})', color='red')
    ax2.axvline(x=0, color='black', linestyle='--')
    ax2.set_xlabel('Prediction Error (°C)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Time series comparison
    ax3 = axes[1, 0]
    x_range = range(len(y_test))
    ax3.plot(x_range, y_test, 'k-', label='Actual', linewidth=2)
    ax3.plot(x_range, lr_pred, 'b--', label='Linear Regression', alpha=0.7)
    ax3.plot(x_range, nn_pred, 'r--', label='Neural Network', alpha=0.7)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Temperature Predictions Over Test Set')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Metrics comparison bar chart
    ax4 = axes[1, 1]
    metrics = ['MSE', 'R² Score', 'Within 1°C (%)', 'Within 2°C (%)']
    lr_within_1 = np.mean(np.abs(y_test - lr_pred) <= 1) * 100
    lr_within_2 = np.mean(np.abs(y_test - lr_pred) <= 2) * 100
    nn_within_1 = np.mean(np.abs(y_test - nn_pred) <= 1) * 100
    nn_within_2 = np.mean(np.abs(y_test - nn_pred) <= 2) * 100

    lr_values = [lr_mse, lr_r2 * 100, lr_within_1, lr_within_2]
    nn_values = [nn_mse, nn_r2 * 100, nn_within_1, nn_within_2]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax4.bar(x - width/2, lr_values, width, label='Linear Regression', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, nn_values, width, label='Neural Network', color='red', alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Model Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'model_comparison.png', dpi=150)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Linear Regression':>20} {'Neural Network':>20}")
    print("-" * 60)
    print(f"{'MSE':<20} {lr_mse:>20.2f} {nn_mse:>20.2f}")
    print(f"{'R² Score':<20} {lr_r2:>20.2f} {nn_r2:>20.2f}")
    print(f"{'Within 1°C':<20} {lr_within_1:>19.1f}% {nn_within_1:>19.1f}%")
    print(f"{'Within 2°C':<20} {lr_within_2:>19.1f}% {nn_within_2:>19.1f}%")
    print("\nComparison graph saved to: model_comparison.png")


def main():
    print("Loading data...")
    X_train, y_train, X_test, y_test, test_dates = load_data()

    print("Training Linear Regression...")
    lr_predictions = train_linear_regression(X_train, y_train, X_test)

    print("Training Neural Network...")
    nn_predictions = train_neural_network(X_train, y_train, X_test)

    print("Saving predictions to CSV...")
    save_predictions_csv(y_test, lr_predictions, nn_predictions, test_dates)

    print("Generating comparison plots...")
    plot_comparison(y_test, lr_predictions, nn_predictions)


if __name__ == "__main__":
    main()
