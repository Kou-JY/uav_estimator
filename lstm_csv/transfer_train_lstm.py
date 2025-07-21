import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lstm_csv.model import LSTMModel
import joblib

# ===== 参数配置 =====
DATA_DIR = "../data/my_csv_segments"
MODEL_DIR = "../models/Transfer_lstm_10"
PRETRAINED_MODEL = "../models/ETH_lstm_10/lstm_model.pt"
WINDOW_SIZE = 100
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset 类 =====
class FlightDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ===== 数据加载函数 =====
def load_flight_data(flight_ids):
    X_all, Y_all = [], []
    for i in flight_ids:
        df = pd.read_csv(os.path.join(DATA_DIR, f"Flight_my_{i:02d}.csv"))
        df = df.dropna()
        X = df[['Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator',
                'Pitch_rate', 'Roll_rate', 'Rudder', 'Yaw_rate']].values
        Y = df['AOA'].values
        for j in range(len(X) - WINDOW_SIZE):
            X_all.append(X[j:j+WINDOW_SIZE])
            Y_all.append(Y[j+WINDOW_SIZE])
    return np.array(X_all), np.array(Y_all)

# ===== 微调函数 =====
def transfer_train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, Y_train = load_flight_data([1, 2])
    X_test, Y_test = load_flight_data([3])

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    Y_train_scaled = scaler_y.fit_transform(Y_train.reshape(-1, 1)).reshape(-1)
    X_test_scaled = scaler_x.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
    Y_test_scaled = scaler_y.transform(Y_test.reshape(-1, 1)).reshape(-1)

    joblib.dump(scaler_x, os.path.join(MODEL_DIR, "transfer_input_scaler.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, "transfer_aoa_scaler.pkl"))

    train_loader = DataLoader(FlightDataset(X_train_scaled, Y_train_scaled), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FlightDataset(X_test_scaled, Y_test_scaled), batch_size=BATCH_SIZE)

    model = LSTMModel(input_dim=10).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=DEVICE))

    # ===== 冻结前面层（可选）
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                output = model(X_batch)
                val_loss += criterion(output, Y_batch).item()
        avg_val_loss = val_loss / len(test_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transfer_lstm_model.pt"))
            print("模型已保存")

if __name__ == '__main__':
    transfer_train()
