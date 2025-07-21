
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from simpleTCN import SimpleTCN
from train_TCN_matlab import FlightDataset, load_flight_data
import joblib

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 12
r2d = 57.295779513
# ========== 加载模型 ==========
# model = SimpleTCN(input_dim=4, num_blocks=4, num_filters=64, kernel_size=3, dropout=0.2).to(DEVICE)     #for ETH data
# model.load_state_dict(torch.load("tcn_model_matlab_aligned.pt", map_location=DEVICE))
model = SimpleTCN(input_dim=4, num_blocks=2, num_filters=32, kernel_size=3, dropout=0.1).to(DEVICE)     #for my uav data
model.load_state_dict(torch.load("../models/MY_tcn_4/my_tcn_model.pt", map_location=DEVICE))
model.eval()

# ========== 加载归一化器 ==========
scaler_x = joblib.load("../models/MY_tcn_4/scaler_x.pkl")
scaler_y = joblib.load("../models/MY_tcn_4/scaler_y.pkl")

# ========== 预处理函数 ==========
def preprocess(flight_ids):
    X_raw, Y_raw = load_flight_data(flight_ids)
    X_scaled = scaler_x.transform(X_raw.reshape(-1, X_raw.shape[2])).reshape(X_raw.shape)
    Y_scaled = scaler_y.transform(Y_raw.reshape(-1, 1)).reshape(-1)  # [N]
    return X_scaled, Y_scaled

# ========== 测试并可视化 ==========
# def evaluate_and_plot(flight_ids):
#     for flight_id in flight_ids:
#         X_test, Y_test = preprocess([flight_id])
#         test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=1, shuffle=False)
#
#         y_true, y_pred = [], []
#
#         with torch.no_grad():
#             for X_batch, Y_batch in test_loader:
#                 X_batch = X_batch.to(DEVICE)
#                 Y_batch = Y_batch.to(DEVICE)
#                 pred = model(X_batch)[:, -1]  # 只取最后一个时间步输出 [B]
#                 y_pred.extend(pred.cpu().numpy())
#                 y_true.extend(Y_batch.cpu().numpy())
#
#         y_true = np.array(y_true).reshape(-1, 1)
#         y_pred = np.array(y_pred).reshape(-1, 1)
#
#         # 反归一化
#         y_true = scaler_y.inverse_transform(y_true).reshape(-1)
#         y_pred = scaler_y.inverse_transform(y_pred).reshape(-1)
#
#         # 计算指标
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#         mae = mean_absolute_error(y_true, y_pred)
#
#         print(f"Flight {flight_id} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
#
#         # 可视化
#         plt.figure(figsize=(10, 5))
#         plt.plot(y_true, label="True AOA")
#         plt.plot(y_pred, label="Predicted AOA")
#         plt.title(f"Flight {flight_id} - Final Step AOA Prediction")
#         plt.xlabel("Time Steps")
#         plt.ylabel("AOA (degrees)")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

def evaluate_and_plot(flight_ids):
    all_rmses = []

    for flight_id in flight_ids:
        X_test, Y_test = preprocess([flight_id])
        test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=1, shuffle=False)

        y_true, y_pred = [], []

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                pred = model(X_batch)[:, -1]  # 只取最后一个时间步
                y_pred.extend(pred.cpu().numpy())
                y_true.extend(Y_batch.cpu().numpy())

        y_true = np.array(y_true).reshape(-1, 1)
        y_pred = np.array(y_pred).reshape(-1, 1)

        y_true = scaler_y.inverse_transform(y_true).reshape(-1)
        y_pred = scaler_y.inverse_transform(y_pred).reshape(-1)

        # 计算误差
        rmse = r2d*np.sqrt(mean_squared_error(y_true, y_pred))
        mae = r2d*mean_absolute_error(y_true, y_pred)
        all_rmses.append(rmse)

        print(f"Flight {flight_id} - RMSE: {rmse:.4f}(deg), MAE: {mae:.4f}(deg)")

        # 可视化
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label="True AOA")
        plt.plot(y_pred, label="Predicted AOA")
        plt.title(f"Flight {flight_id} - AOA Prediction")
        plt.xlabel("Time Step")
        plt.ylabel("AOA (degrees)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 汇总平均 RMSE
    mean_rmse = np.mean(all_rmses)
    print("Summary RMSE over all flights:")
    for i, rmse in zip(flight_ids, all_rmses):
        print(f"  Flight {i}: {rmse:.4f}")
    print(f"Mean RMSE across flights: {mean_rmse:.4f}")
# ========== 执行测试 ==========
# evaluate_and_plot([15, 16, 17, 18])
evaluate_and_plot([3])


















'''
# ✅ test_tcn.py - 测试脚本（带 RMSE/MAE + 归一化器加载）
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from TCN_csv.simpleTCN import SimpleTCN
from TCN_csv.train_TCN_matlab import FlightDataset, load_flight_data
import joblib

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 12

# ========== 加载模型 ==========
model = SimpleTCN(input_dim=4, num_blocks=4, num_filters=64, kernel_size=3, dropout=0.2).to(DEVICE)
model.load_state_dict(torch.load("tcn_model_matlab_aligned.pt", map_location=DEVICE))
model.eval()

# ========== 加载归一化器 ==========
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ========== 预处理函数 ==========
def preprocess(flight_ids):
    X_raw, Y_raw = load_flight_data(flight_ids)
    X_scaled = scaler_x.transform(X_raw.reshape(-1, X_raw.shape[2])).reshape(X_raw.shape)
    #Y_scaled = scaler_y.transform(Y_raw.reshape(-1, 1)).reshape(-1)
    Y_scaled = scaler_y.transform(Y_raw.reshape(-1, 1)).reshape(Y_raw.shape)
    return X_scaled, Y_scaled

# ========== 测试并可视化 ==========
def evaluate_and_plot(flight_ids):
    for flight_id in flight_ids:
        X_test, Y_test = preprocess([flight_id])
        test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=1, shuffle=False)

        y_true, y_pred = [], []

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                pred_seq = model(X_batch).cpu().numpy().squeeze(0)  # [T]
                y_pred.append(pred_seq.reshape(-1))  # 保证是1维
                y_true.append(Y_batch.numpy().reshape(-1))  # 保证是1维
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                pred_seq = model(X_batch).cpu().numpy() # [B, T]
                y_pred.append(pred_seq)  # -> [B*T]
                y_true.append(Y_batch.cpu().numpy())

        y_true = np.concatenate(y_true, axis=0).reshape(-1)
        y_pred = np.concatenate(y_pred, axis=0).reshape(-1)
        print("y_true shape:", y_true.shape)
        print("y_pred shape:", y_pred.shape)

        # 反归一化
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"Flight {flight_id} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # 可视化
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label="True AOA")
        plt.plot(y_pred, label="Predicted AOA")
        plt.title(f"Flight {flight_id} - Sequence Prediction")
        plt.xlabel("Time Steps")
        plt.ylabel("AOA (degrees)")
        plt.legend()
        plt.grid(True)
        plt.show()


# ========== 执行测试 ==========
evaluate_and_plot([15, 16, 17, 18])'''





















'''import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from TCN_csv.simpleTCN import SimpleTCN
from TCN_csv.train_TCN_matlab import FlightDataset, load_flight_data
import joblib

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 12

# ========== 加载模型 ==========
model = SimpleTCN(input_dim=4, num_blocks=4, num_filters=64, kernel_size=3, dropout=0.2).to(DEVICE)
model.load_state_dict(torch.load("tcn_model_matlab_aligned.pt", map_location=DEVICE))
model.eval()

# ========== 加载归一化器 ==========
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ========== 预处理函数 ==========
def preprocess(flight_ids):
    X_raw, Y_raw = load_flight_data(flight_ids)
    X_scaled = scaler_x.transform(X_raw.reshape(-1, X_raw.shape[2])).reshape(X_raw.shape)
    Y_scaled = scaler_y.transform(Y_raw.reshape(-1, 1)).reshape(-1)
    return X_scaled, Y_scaled

# ========== 测试并可视化 ==========
def evaluate_and_plot(flight_ids):
    for flight_id in flight_ids:
        X_test, Y_test = preprocess([flight_id])
        test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=1, shuffle=False)

        y_true, y_pred = [], []

        with torch.no_grad():
                #for X_batch, Y_batch in test_loader:
                # X_batch = X_batch.to(DEVICE)
                #pred = model(X_batch).cpu().numpy()
                #y_pred.extend(pred)
                #y_true.extend(Y_batch.numpy())
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                pred_seq = model(X_batch).cpu().numpy()  # [1, T]
                pred = pred_seq[:, -1]  # 只取最后一个时间步
                y_pred.extend(pred)
                y_true.extend(Y_batch.numpy())
        # 反归一化
        y_true = scaler_y.inverse_transform(np.array(y_true).reshape(-1, 1)).reshape(-1)
        y_pred = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).reshape(-1)

        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        print(f"Flight {flight_id} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # 可视化
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label="True AOA")
        plt.plot(y_pred, label="Predicted AOA")
        plt.title(f"Flight {flight_id} - AOA Prediction")
        plt.xlabel("Time Steps")
        plt.ylabel("AOA (degrees)")
        plt.legend()
        plt.grid(True)
        plt.show()

# ========== 运行测试 ==========
evaluate_and_plot([15, 16, 17, 18])'''
