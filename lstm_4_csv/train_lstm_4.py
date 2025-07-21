import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========== 配置 ==========
DATA_DIR = r"D:\\code\\uav_angle_estimator\\data\\my_csv_segments"

WINDOW_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据集定义 ==========
class FlightDataset(Dataset):
    def __init__(self, X_seqs, Y_vals):
        self.X = torch.tensor(X_seqs, dtype=torch.float32)
        self.Y = torch.tensor(Y_vals, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ========== 模型定义 ==========
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):

        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # 取最后一个时间步输出
        out = self.fc(out)
        return out.squeeze(-1)

# ========== 数据预处理 ==========
def load_flight_data(flight_ids):
    X_all, Y_all = [], []
    for i in flight_ids:
        df = pd.read_csv(os.path.join(DATA_DIR, f"Flight_my_{i:02d}.csv"))
        X = df[['Acc_z', 'Airspeed', 'Elevator', 'Pitch_rate']].values
        Y = df['AOA'].values

        for j in range(len(X) - WINDOW_SIZE):
            X_all.append(X[j:j + WINDOW_SIZE])
            Y_all.append(Y[j + WINDOW_SIZE])

    return np.array(X_all), np.array(Y_all)

def preprocess():
    # X_train_raw, Y_train_raw = load_flight_data(range(1, 15))
    # X_test_raw, Y_test_raw = load_flight_data(range(15, 19))
    X_train_raw, Y_train_raw = load_flight_data([1, 2])
    X_test_raw, Y_test_raw = load_flight_data([3])

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_raw.reshape(-1, X_train_raw.shape[2]))
    X_train_scaled = X_train_scaled.reshape(X_train_raw.shape)
    Y_train_scaled = scaler_y.fit_transform(Y_train_raw.reshape(-1, 1)).reshape(-1)

    X_test_scaled = scaler_x.transform(X_test_raw.reshape(-1, X_test_raw.shape[2]))
    X_test_scaled = X_test_scaled.reshape(X_test_raw.shape)
    Y_test_scaled = scaler_y.transform(Y_test_raw.reshape(-1, 1)).reshape(-1)

    # 获取 train_lstm.py 的绝对路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 指向 ../models/MY_lstm_10
    SAVE_DIR = os.path.join(BASE_DIR, "..", "models", "MY_lstm_4")
    os.makedirs(SAVE_DIR, exist_ok=True)

    np.save(os.path.join(SAVE_DIR, "my_input_mean.npy"), scaler_x.mean_)
    np.save(os.path.join(SAVE_DIR, "my_input_std.npy"), scaler_x.scale_)
    np.save(os.path.join(SAVE_DIR, "my_aoa_mean.npy"), scaler_y.mean_)
    np.save(os.path.join(SAVE_DIR, "my_aoa_std.npy"), scaler_y.scale_)

    return X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled

# ========== 训练函数 ==========
def train():
    X_train, Y_train, X_test, Y_test = preprocess()
    train_loader = DataLoader(FlightDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(input_dim=4).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            #print(X_batch.shape)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "MY_lstm_4")
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "my_lstm_model.pt")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("模型已保存")

if __name__ == '__main__':
    train()
