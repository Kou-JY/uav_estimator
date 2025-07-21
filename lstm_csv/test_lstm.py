import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from model import LSTMModel  # 假设你的模型保存在 lstm_model.py 文件中
from train_lstm import FlightDataset, load_flight_data  # 从训练脚本中导入 FlightDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

r2d = 57.295779513
# ========== 数据加载与标准化 ==========
def preprocess(flight_ids, scaler_x, scaler_y):
    X_raw, Y_raw = load_flight_data(flight_ids)

    # 使用训练阶段的标准化参数
    X_scaled = scaler_x.transform(X_raw.reshape(-1, X_raw.shape[2])).reshape(X_raw.shape)
    Y_scaled = scaler_y.transform(Y_raw.reshape(-1, 1)).reshape(-1)

    return X_scaled, Y_scaled


# ========== 模型加载与预测 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LSTMModel(input_dim=10).to(device)
model = LSTMModel(input_dim=10, hidden_dim=32).to(device)   #hidden_dim=64对应ETH数据
model.load_state_dict(torch.load("../models/MY_lstm_10/my_lstm_model.pt"))
model.eval()


# ========== 预测与可视化 ==========
def visualize_predictions(flight_ids, scaler_x, scaler_y):
    for flight_id in flight_ids:
        X_test, Y_test = preprocess([flight_id], scaler_x, scaler_y)  # 使用修正后的 preprocess
        test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=1, shuffle=False)

        y_pred = []
        y_true = []

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_true.extend(Y_batch.numpy())

                pred = model(X_batch).cpu().numpy()
                y_pred.extend(pred)

        # 反标准化
        y_true = scaler_y.inverse_transform(np.array(y_true).reshape(-1, 1)).reshape(-1)
        y_pred = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).reshape(-1)

        rmse = r2d*np.sqrt(mean_squared_error(y_true, y_pred))
        mae = r2d*mean_absolute_error(y_true, y_pred)
        all_rmses = []
        all_rmses.append(rmse)

        print(f"Flight {flight_id} - RMSE: {rmse:.4f}(deg), MAE: {mae:.4f}(deg)")

        # 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label="True AOA")
        plt.plot(y_pred, label="Predicted AOA")
        plt.title(f"Flight {flight_id} AOA Prediction vs True Values")
        plt.xlabel("Time Steps")
        plt.ylabel("AOA (degrees)")
        plt.legend()
        plt.grid(True)
        plt.show()


# ========== 加载标准化参数 ==========
scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaler_x.mean_ = np.load("../models/MY_lstm_10/my_input_mean.npy")
scaler_x.scale_ = np.load("../models/MY_lstm_10/my_input_std.npy")
scaler_y.mean_ = np.load("../models/MY_lstm_10/my_aoa_mean.npy")
scaler_y.scale_ = np.load("../models/MY_lstm_10/my_aoa_std.npy")

# 进行可视化
# visualize_predictions([15, 16, 17, 18], scaler_x, scaler_y)
visualize_predictions([3], scaler_x, scaler_y)