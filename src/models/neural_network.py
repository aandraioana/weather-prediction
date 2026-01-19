import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).parent.parent.parent

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

HIDDEN_LAYERS = [128, 64, 32]
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32
DROPOUT_RATE = 0.1
PATIENCE = 50  # Early stopping patience
VAL_SPLIT = 0.2  # Validation split ratio

# =========================
# Model
# =========================

class WeatherPredictor(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.0):
        super().__init__()

        layers = []
        prev = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = h

        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================
# Data Loader
# =========================

def load_dataset(folder, target_cols, drop_cols):

    base = PROJECT_ROOT / f"data/{folder}"

    train_df = pd.read_csv(base / "train.csv")
    test_df = pd.read_csv(base / "test.csv")

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    if "season" in train_df.columns:
        season_map = {"winter":1, "spring":2, "summer":3, "fall":4, "autumn":4}
        train_df["season"] = train_df["season"].str.lower().map(season_map)
        test_df["season"] = test_df["season"].str.lower().map(season_map)

    feature_cols = [
        c for c in train_df.columns
        if c not in ["date", "location_id"] + target_cols + drop_cols
    ]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_cols].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[target_cols].values

    return X_train, y_train, X_test, y_test, feature_cols

# =========================
# Training & Evaluation
# =========================

def run_experiment(name, folder, targets, drop_cols):

    print("\n" + "="*70)
    print(f"Dataset: {name}")
    print("="*70)

    X_train_full, y_train_full, X_test, y_test, features = load_dataset(
        folder, targets, drop_cols
    )

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SPLIT, random_state=SEED
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples  : {len(X_val)}")
    print(f"Test samples : {len(X_test)}")
    print(f"Features     : {len(features)}")
    print(f"Targets      : {targets}")

    # Scale features and targets
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    # Create data loaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_s),
        torch.FloatTensor(y_train_s)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val_s),
        torch.FloatTensor(y_val_s)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = WeatherPredictor(
        input_size=X_train_s.shape[1],
        hidden_layers=HIDDEN_LAYERS,
        output_size=y_train.shape[1],
        dropout_rate=DROPOUT_RATE
    )

    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )

    print("\nTraining...")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch+1) % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {lr:.6f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test_s)).detach().cpu().tolist()
        preds = np.array(preds)

    preds_orig = scaler_y.inverse_transform(preds)
    y_test_orig = scaler_y.inverse_transform(y_test_s)

    mse = mean_squared_error(y_test_orig, preds_orig)
    r2 = r2_score(y_test_orig, preds_orig)

    print("\nResults:")
    print(f"MSE: {mse:.2f}")
    print(f"R² : {r2:.4f}")

    # Per-target metrics with appropriate thresholds
    for i, t in enumerate(targets):
        y_true = y_test_orig[:, i]
        y_pred = preds_orig[:, i]
        target_range = y_true.max() - y_true.min()
        target_std = y_true.std()

        # Use percentage of range for accuracy thresholds
        thresh_5pct = target_range * 0.05
        thresh_10pct = target_range * 0.10

        acc_5 = np.mean(np.abs(y_true - y_pred) <= thresh_5pct) * 100
        acc_10 = np.mean(np.abs(y_true - y_pred) <= thresh_10pct) * 100

        target_r2 = r2_score(y_true, y_pred)
        target_mse = mean_squared_error(y_true, y_pred)

        print(f"\n{t}:")
        print(f"  R²: {target_r2:.4f} | MSE: {target_mse:.2f}")
        print(f"  Accuracy ±5%: {acc_5:.1f}% | ±10%: {acc_10:.1f}%")

# =========================
# Main
# =========================

def main():

    run_experiment(
        name="Savanna Preserve",
        folder="savanna_preserve",
        targets=["temperature_2m"],
        drop_cols=["relative_humidity_2m"]
    )

    # Urban Air dataset skipped - only 168 samples with poor train/test split

    run_experiment(
        name="Resilient Fields - Irradiance",
        folder="resilient_fields",
        targets=["global_tilted_irradiance"],
        drop_cols=["precipitation"]
    )

    run_experiment(
        name="Resilient Fields - Precipitation",
        folder="resilient_fields",
        targets=["precipitation"],
        drop_cols=["global_tilted_irradiance"]
    )

if __name__ == "__main__":
    main()
