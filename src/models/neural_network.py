import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

VAL_SPLIT = 0.2

CONFIGS = {
    "savanna_temperature": {
        "hidden_layers": [128, 64, 32],
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "dropout_rate": 0.1,
        "patience": 50,
        "use_log_target": False,
    },
    "urban_aqi": {
        "hidden_layers": [128, 64, 32],
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "dropout_rate": 0.1,
        "patience": 50,
        "use_log_target": False,
    },
    "resilient_irradiance": {
        "hidden_layers": [256, 128, 64],
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 32,
        "dropout_rate": 0.2,
        "patience": 50,
        "use_log_target": False,
    },
    "resilient_precipitation": {
        "hidden_layers": [32],
        "learning_rate": 0.001,
        "epochs": 2000,
        "batch_size": 32,
        "dropout_rate": 0.0,
        "patience": 150,
        "use_log_target": False,
    },
}


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
# Preprocessing
# =========================

def add_cyclical_encoding(df):
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df.drop("month", axis=1, inplace=True)

    if "season" in df.columns:
        if df["season"].dtype == object:
            season_map = {"winter":1, "spring":2, "summer":3, "fall":4, "autumn":4}
            df["season"] = df["season"].str.lower().map(season_map)
        df["season_sin"] = np.sin(2 * np.pi * df["season"] / 4)
        df["season_cos"] = np.cos(2 * np.pi * df["season"] / 4)
        df.drop("season", axis=1, inplace=True)

    return df


def add_lag_differences(df, prefix):
    lag_cols = [c for c in df.columns if f"{prefix}_previous_day" in c]
    lag_cols = sorted(lag_cols, key=lambda x: int(x.split("day")[-1]))

    if len(lag_cols) >= 2:
        df[f"{prefix}_change_1d"] = df[lag_cols[0]] - df[lag_cols[1]]
    if len(lag_cols) >= 7:
        df[f"{prefix}_change_7d"] = df[lag_cols[0]] - df[lag_cols[6]]
        df[f"{prefix}_std_7d"] = df[lag_cols].std(axis=1)

    return df


def preprocess_features(df):
    df = df.copy()
    df = add_cyclical_encoding(df)

    for col in df.columns:
        if "_previous_day1" in col:
            prefix = col.replace("_previous_day1", "")
            df = add_lag_differences(df, prefix)

    return df


# =========================
# Dataset
# =========================

def load_dataset(folder, target_cols, drop_cols, use_log_target=False):
    base = PROJECT_ROOT / f"data/{folder}"

    train_df = pd.read_csv(base / "train.csv")
    test_df = pd.read_csv(base / "test.csv")

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    train_df = preprocess_features(train_df)
    test_df = preprocess_features(test_df)

    feature_cols = [
        c for c in train_df.columns
        if c not in ["date", "location_id"] + target_cols + drop_cols
    ]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_cols].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[target_cols].values

    if use_log_target:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    return X_train, y_train, X_test, y_test, feature_cols, use_log_target


# =========================
# Plotting
# =========================

def plot_training_curves(train_losses, val_losses, name):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Training Curves – {name}")
    plt.legend()
    plt.grid(alpha=0.3)

    out = PROJECT_ROOT / f"training_curve_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_predictions(y_true, y_pred, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual – {name}")
    plt.grid(alpha=0.3)

    out = PROJECT_ROOT / f"pred_vs_actual_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_error_hist(y_true, y_pred, name):
    errors = y_true - y_pred
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=40, alpha=0.75)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution – {name}")
    plt.grid(alpha=0.3)

    out = PROJECT_ROOT / f"error_hist_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# =========================
# Training
# =========================

def run_experiment(name, folder, targets, drop_cols, config_name):
    cfg = CONFIGS[config_name]

    X_train_full, y_train_full, X_test, y_test, features, use_log = load_dataset(
        folder, targets, drop_cols, use_log_target=cfg['use_log_target']
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SPLIT, random_state=SEED
    )

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train_s)),
        batch_size=cfg['batch_size'], shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val_s)),
        batch_size=cfg['batch_size']
    )

    model = WeatherPredictor(X_train_s.shape[1], cfg['hidden_layers'], y_train.shape[1], cfg['dropout_rate'])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

    train_losses, val_losses = [], []

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(cfg['epochs']):
        model.train()
        t_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                v_loss += criterion(model(xb), yb).item()

        t_loss /= len(train_loader)
        v_loss /= len(val_loader)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        scheduler.step(v_loss)

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg['patience']:
            break

    model.load_state_dict(best_state)

    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test_s)).numpy()

    preds_orig = scaler_y.inverse_transform(preds)
    y_test_orig = scaler_y.inverse_transform(y_test_s)

    if use_log:
        preds_orig = np.expm1(preds_orig)
        y_test_orig = np.expm1(y_test_orig)

    mse = mean_squared_error(y_test_orig, preds_orig)
    r2 = r2_score(y_test_orig, preds_orig)

    print(f"\n{name} → MSE: {mse:.2f} | R²: {r2:.4f}")

    plot_training_curves(train_losses, val_losses, name)
    plot_predictions(y_test_orig[:, 0], preds_orig[:, 0], name)
    plot_error_hist(y_test_orig[:, 0], preds_orig[:, 0], name)


def main():
    run_experiment(
        "Savanna Preserve - Temperature",
        "savanna_preserve",
        ["temperature_2m"],
        ["relative_humidity_2m"],
        "savanna_temperature"
    )

    run_experiment(
        "Clean Urban Air - AQI",
        "clean_urban_air",
        ["us_aqi"],
        ["relative_humidity_2m"],
        "urban_aqi"
    )

    run_experiment(
        "Resilient Fields - Irradiance",
        "resilient_fields",
        ["global_tilted_irradiance"],
        ["precipitation"],
        "resilient_irradiance"
    )

    run_experiment(
        "Resilient Fields - Precipitation",
        "resilient_fields",
        ["precipitation"],
        ["global_tilted_irradiance"],
        "resilient_precipitation"
    )


if __name__ == "__main__":
    main()
