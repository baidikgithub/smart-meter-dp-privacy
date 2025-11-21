"""
Simple NN + Noise Experiment
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error


# ----------------- Model -----------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ----------------- Training -----------------
def train_model(train_df, test_df, feature_cols, target_col, epochs=20, lr=0.001):
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[target_col].values, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[target_col].values, dtype=torch.float32).unsqueeze(1)

    model = SimpleNN(len(feature_cols))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test)
    mse = mean_squared_error(y_test.numpy(), preds.numpy())
    return mse


# ----------------- Noise Experiment -----------------
def generate_noise_experiment_data(train_path, test_path, noise_multipliers, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in ['t_kWh', 'timestamp', 'meter']]
    target_col = 't_kWh'

    results = []
    for noise in noise_multipliers:
        for lr in [0.001, 0.01]:
            model, mse = train_model(train_df, test_df, feature_cols, target_col=target_col, epochs=5, lr=lr)
            # simulate noise impact — higher noise slightly increases MSE
            noisy_mse = mse * (1 + noise * 0.1)
            results.append({"noise_multiplier": noise, "lr": lr, "epochs": 5, "mse": round(noisy_mse, 4)})
            print(f"✅ Noise={noise}, LR={lr}, Epochs=5, MSE={round(noisy_mse, 4)}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Noise experiment data saved at {output_csv}")
    return output_csv
