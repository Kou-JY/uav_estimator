# ✅ train_tcn.py - 与 MATLAB 结构对齐的主训练脚本
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from tcn_csv.simpleTCN  import SimpleTCN
import joblib
# ========== 配置 ==========
# 配置参数
DATA_DIR = r"D:\\code\\uav_angle_estimator\\data\\my_csv_segments"
WINDOW_SIZE = 100
EPOCHS = 50
# LEARNING_RATE = 0.005   #for ETH data
# BATCH_SIZE = 128        #for ETH data
LEARNING_RATE = 0.001     #for my uav data
BATCH_SIZE = 32           #for my uav data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 自定义 Dataset
class FlightDataset(Dataset):
    def __init__(self, X_seqs, Y_vals):
        self.X = torch.tensor(X_seqs, dtype=torch.float32)
        self.Y = torch.tensor(Y_vals, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 加载并预处理数据

def load_flight_data(flight_ids):
    X_all, Y_all = [], []
    for i in flight_ids:
        df = pd.read_csv(os.path.join(DATA_DIR, f"Flight_my_{i:02d}.csv"))
        X = df[['Acc_z', 'Airspeed', 'Elevator', 'Pitch_rate']].values
        Y = df['AOA'].values

        for j in range(len(X) - WINDOW_SIZE):
            X_all.append(X[j:j + WINDOW_SIZE])
            Y_all.append(Y[j + WINDOW_SIZE - 1])  # 取窗口最后一个时间步的目标值

    return np.array(X_all), np.array(Y_all)

def preprocess():
    # X_train_raw, Y_train_raw = load_flight_data(range(1, 15))
    # X_test_raw, Y_test_raw = load_flight_data(range(15, 19))
    X_train_raw, Y_train_raw = load_flight_data([1, 2])
    X_test_raw, Y_test_raw = load_flight_data([3])

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_raw.reshape(-1, X_train_raw.shape[2]))
    X_train_scaled = X_train_scaled.reshape(X_train_raw.shape)
    Y_train_scaled = scaler_y.fit_transform(Y_train_raw.reshape(-1, 1)).reshape(-1)

    X_test_scaled = scaler_x.transform(X_test_raw.reshape(-1, X_test_raw.shape[2]))
    X_test_scaled = X_test_scaled.reshape(X_test_raw.shape)
    Y_test_scaled = scaler_y.transform(Y_test_raw.reshape(-1, 1)).reshape(-1)

    joblib.dump(scaler_x, "../models/MY_tcn_4/scaler_x.pkl")
    joblib.dump(scaler_y, "../models/MY_tcn_4/scaler_y.pkl")

    return X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_y

# 训练主函数
def train():
    X_train, Y_train, X_test, Y_test, scaler_y = preprocess()
    train_loader = DataLoader(FlightDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    # model = SimpleTCN(input_dim=4, num_blocks=4, num_filters=64, kernel_size=3, dropout=0.2).to(DEVICE)     #for ETH data
    model = SimpleTCN(input_dim=4, num_blocks=2, num_filters=32, kernel_size=3, dropout=0.1).to(DEVICE)      #for my uav data
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    best_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in tqdm(range(EPOCHS), desc="训练进度"):
        model.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(X_batch)[:, -1]  # 取最后一个时间步的预测值 [B]
                loss = criterion(outputs, Y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                outputs = model(X_batch)[:, -1]
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "MY_tcn_4")
            os.makedirs(MODEL_DIR, exist_ok=True)
            MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "my_tcn_model.pt")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            counter = 0
            print("模型改进，已保存。")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

if __name__ == '__main__':
    train()







'''DATA_DIR = r"E:\\tiaozhanbei\\code\\python\\uav_angle_estimator\\data\\csv_segments"
WINDOW_SIZE = 12
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ========== Dataset 定义 ==========
class FlightDataset(Dataset):
    def __init__(self, X_seqs, Y_vals):
        self.X = torch.tensor(X_seqs, dtype=torch.float32)
        self.Y = torch.tensor(Y_vals, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ========== 数据加载与预处理 ==========
def load_flight_data(flight_ids):
    X_all, Y_all = [], []
    for i in flight_ids:
        df = pd.read_csv(os.path.join(DATA_DIR, f"Flight_{i:02d}.csv"))
        X = df[['Acc_z', 'Airspeed', 'Elevator', 'Pitch_rate']].values
        Y = df['AOA'].values
        #Y_seq = []

        for j in range(len(X) - WINDOW_SIZE):
            X_all.append(X[j:j + WINDOW_SIZE])
            #Y_all.append(Y[j + WINDOW_SIZE])
            Y_all.append(Y[j :j + WINDOW_SIZE])

    return np.array(X_all), np.array(Y_all)

def preprocess():
    X_train_raw, Y_train_raw = load_flight_data(range(1, 15))
    X_test_raw, Y_test_raw = load_flight_data(range(15, 19))

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_raw.reshape(-1, X_train_raw.shape[2]))
    X_train_scaled = X_train_scaled.reshape(X_train_raw.shape)
    #Y_train_scaled = scaler_y.fit_transform(Y_train_raw.reshape(-1, 1)).reshape(-1)
    Y_train_scaled = scaler_y.fit_transform(Y_train_raw.reshape(-1, 1)).reshape(Y_train_raw.shape)
    Y_test_scaled = scaler_y.transform(Y_test_raw.reshape(-1, 1)).reshape(Y_test_raw.shape)

    X_test_scaled = scaler_x.transform(X_test_raw.reshape(-1, X_test_raw.shape[2]))
    X_test_scaled = X_test_scaled.reshape(X_test_raw.shape)
    #Y_test_scaled = scaler_y.transform(Y_test_raw.reshape(-1, 1)).reshape(-1)
    # ✅ 保存归一化器
    joblib.dump(scaler_x, "scaler_x.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

    return X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_y

# ========== 训练函数 ==========
def train():
    X_train, Y_train, X_test, Y_test, scaler_y = preprocess()
    train_loader = DataLoader(FlightDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleTCN(input_dim=4, num_blocks=4, num_filters=64, kernel_size=3, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    best_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in tqdm(range(EPOCHS), desc="训练进度"):
        model.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                #outputs = model(X_batch)
                #loss = criterion(outputs, Y_batch)
                # 假设输出是 [B, T]
                # 目标 Y 是 [B]，只对应最后一个时间点
                outputs = model(X_batch)  # [B, T]
                #pred_last = outputs[:, -1]  # [B]
                loss = criterion(outputs, Y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                #loss = criterion(outputs.squeeze(-1), Y_batch)

                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "tcn_model_matlab_aligned.pt")
            counter = 0
            print("✅ 模型改进，已保存。")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

if __name__ == '__main__':
    train()'''
