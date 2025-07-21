import os
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from ekf_core import EKFCore
import joblib
import matplotlib.pyplot as plt
import matplotlib

# ===== 模型加载函数 =====
def load_model_and_scaler(model_type, input_dim):
    if model_type == 'lstm_4':
        from models.MY_lstm_4.model import LSTMModel
        model_path = "../models/MY_lstm_4/my_lstm_model.pt"
        x_mean = np.load("../models/MY_lstm_4/my_input_mean.npy")
        x_std = np.load("../models/MY_lstm_4/my_input_std.npy")
        y_mean = np.load("../models/MY_lstm_4/my_aoa_mean.npy")
        y_std = np.load("../models/MY_lstm_4/my_aoa_std.npy")

        model = LSTMModel(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    elif model_type == 'lstm_10':
        from models.MY_lstm_10.model import LSTMModel
        model_path = "../models/MY_lstm_10/my_lstm_model.pt"
        x_mean = np.load("../models/MY_lstm_10/my_input_mean.npy")
        x_std = np.load("../models/MY_lstm_10/my_input_std.npy")
        y_mean = np.load("../models/MY_lstm_10/my_aoa_mean.npy")
        y_std = np.load("../models/MY_lstm_10/my_aoa_std.npy")

        model = LSTMModel(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    elif model_type == 'tcn_4':
        from models.ETH_tcn_4.simpleTCN import SimpleTCN
        model_path = "../models/MY_tcn_4/my_tcn_model.pt"
        with open("../models/MY_tcn_4/scaler_x.pkl", 'rb') as f:
            x_scaler = joblib.load(f)
        with open("../models/MY_tcn_4/scaler_y.pkl", 'rb') as f:
            y_scaler = joblib.load(f)

        model = SimpleTCN(input_dim=4,
                          num_blocks=2,
                          num_filters=32,
                          kernel_size=3,
                          dropout=0.1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        raise ValueError("Unsupported model_type")

    if model_type.startswith("tcn"):
        return model, x_scaler, y_scaler
    else:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_x.mean_, scaler_x.scale_ = x_mean, x_std
        scaler_y.mean_, scaler_y.scale_ = y_mean, y_std
        return model, scaler_x, scaler_y

def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差(MAE)
    """
    # 确保输入为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算绝对值误差并取平均
    absolute_errors = np.abs(y_true - y_pred)
    mae = np.mean(absolute_errors)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return mae, rmse

# ===== 参数设定 =====
flight_id = 3
model_type = 'lstm_10'  # 'lstm_4', 'lstm_10', or 'tcn_4'
GPS_SETTING = 'false'
OBJECT_SETTING = 'alpha'
dt = 1 / 20
g = 32.17
WINDOW_SIZE = 100
r2d = 57.295779513

# ===== 初始化 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dims = {'lstm_4': 4, 'lstm_10': 10, 'tcn_4': 4}
model, scaler_x, scaler_y = load_model_and_scaler(model_type, input_dims[model_type])

# ===== 数据加载 =====
csv_path = os.path.join("../data", "my_csv_segments", f"Flight_my_{flight_id:02d}.csv")
df = pd.read_csv(csv_path)

for col in ['AOA', 'Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator',
            'Pitch_rate', 'Roll_rate', 'Rudder', 'Vd', 'Ve', 'Vn', 'Yaw_rate', 'q0', 'q1', 'q2', 'q3', 'Roll', 'Pitch', 'Yaw']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

# q = df[['q0', 'q1', 'q2', 'q3']].values
# phi, theta, psi = quaternion_to_euler(q)
# psi[psi < 0] += 2 * np.pi
phi = df['Roll'].values.astype(np.float64)
theta = df['Pitch'].values.astype(np.float64)
psi = df['Yaw'].values.astype(np.float64)
ax = (df['Acc_x'].values * 3.2808).astype(np.float64)
ay = (df['Acc_y'].values * 3.2808).astype(np.float64)
az = (df['Acc_z'].values * 3.2808).astype(np.float64)
airspeed = (df['Airspeed'].values * 3.2808).astype(np.float64)
p = df['Roll_rate'].values.astype(np.float64)
q_body = df['Pitch_rate'].values.astype(np.float64)
r = df['Yaw_rate'].values.astype(np.float64)
vn = (df['Vn'].values * 3.2808).astype(np.float64)
ve = (df['Ve'].values * 3.2808).astype(np.float64)
vd = (df['Vd'].values * -3.2808).astype(np.float64)

# initial setting
vt_init = float(airspeed[0])
# alpha_init = float(aoa_true[0])
alpha_init = float(df['AOA'][0])
beta_init = 0.0
if GPS_SETTING == 'false':
    x0 = np.array([vt_init, alpha_init, beta_init], dtype=np.float64)
elif GPS_SETTING == 'true':
    x0 = np.array([vt_init, alpha_init, beta_init, 0, 0, 0], dtype=np.float64)
ekf = EKFCore(GPS_setting=GPS_SETTING, object_setting=OBJECT_SETTING, x_init=x0)

xt_history = []
estimates = []
# N = len(df)
N = len(ax)
print(N)
aoa_pred_list = []

aoa_est_list = []
aoa_true_list = []

# plt.ion()  # 开启交互模式
# fig, plot_ax = plt.subplots()  # 避免与 ax[t] 冲突
# line1, = plot_ax.plot([], [], 'b-', label='Measured AOA')
# line2, = plot_ax.plot([], [], 'r--', label='EKF Estimated AOA')
# plot_ax.set_xlim(0, N)
# plot_ax.set_ylim(min(aoa_true)-0.1, max(aoa_true)+0.1)
# plot_ax.set_xlabel("Time Step")
# plot_ax.set_ylabel("AOA (rad)")
# plot_ax.legend()
# plot_ax.grid(True)

# ===== 主循环 =====
for t in range(N):
    um = np.array([p[t], q_body[t], r[t], ax[t], ay[t], az[t], phi[t], theta[t], psi[t]], dtype=np.float64)
    ekf.predict(um, dt, g)

    if model_type == 'lstm_4' or model_type == 'tcn_4':
        xt = df[['Acc_z', 'Airspeed', 'Elevator', 'Pitch_rate']].iloc[t].values.astype(np.float32)
    else:
        xt = df[['Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator', 'Pitch_rate', 'Roll_rate', 'Rudder', 'Yaw_rate']].iloc[t].values.astype(np.float32)

    xt_scaled = scaler_x.transform(xt.reshape(1, -1)).squeeze(0)
    xt_history.append(xt_scaled)

    # if model_type == 'lstm_4' or model_type == 'lstm_10':
    if len(xt_history) < WINDOW_SIZE:
        # estimates.append(ekf.get_state().copy())
        continue

    input_seq = np.array(xt_history[-WINDOW_SIZE:])
    input_seq = np.expand_dims(input_seq, axis=0)  # (1, 100, dim)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)

    if model_type == 'lstm_4' or model_type == 'lstm_10':
        with torch.no_grad():
            aoa_pred_norm = model(input_tensor).cpu().numpy().item()
        aoa_pred = scaler_y.inverse_transform([[aoa_pred_norm]])[0][0]
    else:
        with torch.no_grad():
            aoa_pred_norm = model(input_tensor)[:, -1].cpu().numpy()  # 只取最后一个时间步输出
        aoa_pred = scaler_y.inverse_transform(aoa_pred_norm.reshape(1, -1))[0, 0]

    aoa_pred_list.append(r2d * aoa_pred)

    if GPS_SETTING == 'false':
        z = np.array([airspeed[t], aoa_pred], dtype=np.float64)
    elif GPS_SETTING == 'true':
        # vn, ve, vd = df['Vn'].iloc[t], df['Ve'].iloc[t], df['Vd'].iloc[t]
        z = np.array([airspeed[t], vn[t], ve[t], vd[t], aoa_pred], dtype=np.float64)

    ekf.update(z, um)
    estimates.append(ekf.get_state().copy())

    # === 实时绘图逻辑 ===
#     aoa_est_list.append(r2d * estimates[-1][1])
#     aoa_true_list.append(aoa_true[t])
#
#     line1.set_data(range(len(aoa_true_list)), aoa_true_list)
#     line2.set_data(range(len(aoa_est_list)), aoa_est_list)
#
#     plot_ax.set_xlim(0, max(100, t))  # 用 plot_ax 而不是 ax
#     plt.pause(0.01)
#
# plt.ioff()
# plt.show()

# ===== 可视化 =====
matplotlib.use('Qt5Agg')  # 或 'QtAgg'，取决于你的环境
estimates = np.array(estimates)

#=====================================================================
window = 50  # 滑动窗口大小
aoa_true_arr = r2d * df['AOA'].values.astype(np.float64)[-len(estimates):]  # 单位：deg（和 estimates 对齐）
aoa_true_smooth = pd.Series(aoa_true_arr).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
aoa_true_std = pd.Series(aoa_true_arr).rolling(window=window, center=True, min_periods=1).std().to_numpy()
aoa_est_arr = aoa_pred_list  # EKF 输出（角度）
aoa_est_smooth = pd.Series(aoa_est_arr).rolling(window=50, center=True, min_periods=1).mean().to_numpy()
aoa_est_std = pd.Series(aoa_est_arr).rolling(window=50, center=True, min_periods=1).std().to_numpy()
#==========================================================================

aoa_true = r2d * df['AOA'].values.astype(np.float64)[-len(estimates):]
mae = calculate_mae(aoa_true, r2d * estimates[:, 1])
# print(f"AOA from EKF RMSE: {rmse:.4f}(deg)")
print(f"AOA from EKF MAE: {mae:.4f}(deg)")
# mae_net = calculate_mae(aoa_true, aoa_pred_list)
# print(f"AOA from Network MAE: {mae_net:.4f}(deg)")

plt.figure(figsize=(12, 6))
plt.plot(aoa_true, label='Measured AOA (deg)', color='blue', alpha=0.7)
plt.plot(r2d * estimates[:, 1], label='EKF Estimated AOA (deg)', color='red', linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("Angle of Attack (deg)")
plt.legend()
plt.grid(True)
plt.title(f"EKF Real-Time AOA Estimation (GPS_SETTING: {GPS_SETTING}, Model: {model_type})")
plt.tight_layout()
plt.show()
#
# plt.figure(figsize=(12, 6))
# # plt.plot(...)
# plt.plot(aoa_true, label='Measured AOA (deg)', color='blue', alpha=0.7)
# plt.plot(aoa_pred_list, label='LSTM Estimated AOA (deg)', color='red', linestyle='--')
# plt.xlabel("Time Step")
# plt.ylabel("Angle of Attack (deg)")
# plt.legend()
# plt.grid(True)
# plt.title(f"LSTM Real-Time AOA Predict (GPS_SETTING: {GPS_SETTING})")
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(12, 6))
# 区域1：真实 AOA ± std
plt.fill_between(range(len(aoa_true_smooth)),
                 aoa_true_smooth - aoa_true_std,
                 aoa_true_smooth + aoa_true_std,
                 color='lightblue', alpha=0.3, label='AOA True ± 1σ')

# 区域2：EKF AOA ± std
plt.fill_between(range(len(aoa_est_smooth)),
                 aoa_est_smooth - aoa_est_std,
                 aoa_est_smooth + aoa_est_std,
                 color='lightcoral', alpha=0.3, label='AOA EKF ± 1σ')

# 曲线：平滑后的真实值
plt.plot(aoa_true_smooth, label='Smoothed Measured AOA (deg)', color='blue')

# 曲线：平滑后的估计值
plt.plot(aoa_est_smooth, label='Smoothed EKF Estimated AOA (deg)', color='red', linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("Angle of Attack (deg)")
plt.legend()
plt.grid(True)
plt.title(f"EKF AOA Estimation with Smoothed ±1σ (Model: {model_type}, GPS: {GPS_SETTING})")
plt.tight_layout()
plt.show()
