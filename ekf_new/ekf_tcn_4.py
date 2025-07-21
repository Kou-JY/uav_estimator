import pandas as pd
import numpy as np
# from models.lstm_model import LSTMModel # Removed as per request
from ekf_core import EKFCore
from units.quaternion import quaternion_to_euler
import os
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence # Removed as per request
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tcn_csv.simpleTCN import SimpleTCN
from sklearn.preprocessing import StandardScaler
from tcn_csv.train_TCN_matlab import FlightDataset, load_flight_data  # 从训练脚本中导入 FlightDataset
#from lstm_csv.test_lstm import preprocess
import joblib

# ========== 数据加载与标准化 ==========
def preprocess(flight_ids, scaler_x, scaler_y):
    X_raw, Y_raw = load_flight_data(flight_ids)

    # 使用训练阶段的标准化参数
    X_scaled = scaler_x.transform(X_raw.reshape(-1, X_raw.shape[2])).reshape(X_raw.shape)
    Y_scaled = scaler_y.transform(Y_raw.reshape(-1, 1)).reshape(-1)
    print(X_raw.shape)
    return X_scaled, Y_scaled


# === Constants ===
g = 32.17  # 重力常数 (ft/s^2)
dt = 1/40  # 采样间隔
input_type = '4aixs' #'allinput','4axis'


# === User-defined settings ===
GPS_SETTING = 'false'  # 'false' or 'true'
OBJECT_SETTING = 'alpha'    # 'alpha' as per request

# === Load CSV Data ===
flight_id = 17
Flight_path = "Flight_" +str(flight_id) +".csv"
csv_path = os.path.join("../data/csv_segments", Flight_path)


df = pd.read_csv(csv_path, on_bad_lines='skip', sep=',') # Explicitly set delimiter

# Convert all relevant columns to numeric, coercing errors to NaN
for col in ['AOA', 'Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator', 'Pitch_rate', 'Roll_rate', 'Rudder', 'Vd', 'Ve', 'Vn', 'Yaw_rate', 'q0', 'q1', 'q2', 'q3']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any NaN values that resulted from coercion (malformed data)
df.dropna(inplace=True)

data = df.values

# Extract quaternion and convert to Euler angles
q = df[["q0", "q1", "q2", "q3"]].values
phi, theta, psi = quaternion_to_euler(q)
psi[psi < 0] += 2 * np.pi  # Map to [0, 2π]

# === Construct input sequences (um) ===
# Ensure all data used for um is converted to float
aileron = df['Aileron'].values.astype(np.float64)
elevator = df['Elevator'].values.astype(np.float64)
rudder = df['Rudder'].values.astype(np.float64)
airspeed = (df['Airspeed'].values * 3.2808).astype(np.float64)  # m/s -> ft/s
ax = (df['Acc_x'].values * 3.2808).astype(np.float64)
ay = (df['Acc_y'].values * 3.2808).astype(np.float64)
az = (df['Acc_z'].values * 3.2808).astype(np.float64)
p = df['Roll_rate'].values.astype(np.float64)
q_body = df['Pitch_rate'].values.astype(np.float64) # Renamed to avoid conflict with quaternion q
r = df['Yaw_rate'].values.astype(np.float64)

# Initial values from the first data point
vt_init = float(airspeed[0])
alpha_init = float(df['AOA'][0]) # Use initial AOA from data as initial guess
beta_init = 0.0 # Assume initial beta is zero

if GPS_SETTING == 'false':
    x0 = np.array([vt_init, alpha_init, beta_init], dtype=np.float64)
elif GPS_SETTING == 'true':
    wn_init = 0.0 # Initial wind biases
    we_init = 0.0
    wd_init = 0.0
    x0 = np.array([vt_init, alpha_init, beta_init, wn_init, we_init, wd_init], dtype=np.float64)
else:
    raise ValueError("Invalid GPS_SETTING.")

ekf = EKFCore(GPS_setting=GPS_SETTING, object_setting=OBJECT_SETTING, x_init=x0)


# 选择设备 (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === 初始化 LSTM 模型 ===
if input_type == 'allinput':
    input_dim = 10
elif input_type == '4aixs':
    input_dim = 4

TCN_model = SimpleTCN(input_dim=4,
    num_blocks=4,
    num_filters=64,
    kernel_size=3,
    dropout=0.2).to(device)
TCN_model.load_state_dict(torch.load("../TCN_csv/tcn_model_matlab_aligned.pt", map_location=device))
TCN_model.eval()



# === Load LSTM Model for AOA Prediction ===

scaler_x = joblib.load("../TCN_csv/scaler_x.pkl")
scaler_y = joblib.load("../TCN_csv/scaler_y.pkl")

aoa_true = df['AOA'].values.astype(np.float64)

# 存储历史数据
aoa_est_list = []
aoa_true_list = []
ext_history = []

WINDOW_SIZE = 100
xt_history = []
estimates = []
N = len(airspeed)
for t in range(N):
    # ==== EKF 输入 um ====
    um = np.array([p[t], q_body[t], r[t], ax[t], ay[t], az[t], phi[t], theta[t], psi[t]], dtype=np.float64)
    ekf.predict(um, dt, g)

    if input_type == 'allinput':
        # ==== 构造 xt (当前时刻10维输入) ====
        xt = np.array([
            df['Acc_x'].iloc[t],
            df['Acc_y'].iloc[t],
            df['Acc_z'].iloc[t],
            df['Aileron'].iloc[t],
            df['Airspeed'].iloc[t],
            df['Elevator'].iloc[t],
            df['Pitch_rate'].iloc[t],
            df['Roll_rate'].iloc[t],
            df['Rudder'].iloc[t],
            df['Yaw_rate'].iloc[t],
        ], dtype=np.float32)

    elif input_type == '4aixs':
        # ==== 构造 xt (当前时刻4维输入) ====
        xt = np.array([
            df['Acc_z'].iloc[t],
            df['Airspeed'].iloc[t],
            df['Elevator'].iloc[t],
            df['Pitch_rate'].iloc[t],
        ], dtype=np.float32)


    # ==== 标准化并放入历史序列 ====
    xt_scaled = scaler_x.transform(xt.reshape(1, -1)).squeeze(0)
    ext_history.append(xt_scaled)

    if len(ext_history) < WINDOW_SIZE:
        continue

    input_seq = np.array(ext_history[-WINDOW_SIZE:])
    input_seq = np.expand_dims(input_seq, axis=0)

    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
    #print(input_tensor.shape)
    #print(input_tensor.shape)
    with torch.no_grad():
        aoa_pred_norm = TCN_model(input_tensor)[:, -1].cpu().numpy() # 只取最后一个时间步输出
    #print(aoa_pred_norm.shape)
    aoa_pred = scaler_y.inverse_transform(aoa_pred_norm.reshape(1, -1))[0, 0]
    aoa_est_list.append(aoa_pred)
    #print(aoa_est_list)

    # ==== 构造 EKF 观测向量 z ====
    if GPS_SETTING == 'false':
        z = np.array([airspeed[t], aoa_pred], dtype=np.float64)
    elif GPS_SETTING == 'true':
        vn = df['Vn'].values.astype(np.float64)[t]
        ve = df['Ve'].values.astype(np.float64)[t]
        vd = df['Vd'].values.astype(np.float64)[t]
        z = np.array([airspeed[t], vn, ve, vd, aoa_pred], dtype=np.float64)
    else:
        raise ValueError("Invalid GPS_SETTING.")

    # ==== EKF 更新 ====
    ekf.update(z, um)
    estimates.append(ekf.get_state().copy())



estimates = np.array(estimates)
valid_aoa_true = aoa_true[WINDOW_SIZE-1:]  # 裁剪掉前100个点


def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差(MAE)

    参数:
    y_true - 真实值的数组
    y_pred - 预测值的数组

    返回:
    mae - 平均绝对误差
    """
    # 确保输入为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算绝对值误差并取平均
    absolute_errors = np.abs(y_true - y_pred)
    mae = np.mean(absolute_errors)

    return mae

mae = calculate_mae(valid_aoa_true, estimates[:, 1])
print(f"AOA MAE: {mae:.4f}(rad)")


# === Visualization ===
plt.figure(figsize=(12, 6))
plt.plot(valid_aoa_true, label='Measured AOA (rad)', color='blue', alpha=0.7)
plt.plot(estimates[:, 1], label='EKF Estimated AOA (rad)', color='red', linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("Angle of Attack (rad)")
plt.legend()
plt.grid(True)
plt.title(f"EKF Real-Time AOA Estimation (GPS_SETTING: {GPS_SETTING})")
plt.tight_layout()
# plt.savefig('aoa_estimation_true_no_h.png')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(aoa_true, label='Measured AOA (rad)', color='blue', alpha=0.7)
plt.plot(aoa_est_list, label='TCN Estimated AOA (rad)', color='red', linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("Angle of Attack (rad)")
plt.legend()
plt.grid(True)
plt.title(f"TCN Real-Time AOA Predict (GPS_SETTING: {GPS_SETTING})")
plt.tight_layout()
# plt.savefig('aoa_estimation_true_no_h.png')
plt.show()

print(f"AOA estimation plot saved to aoa_estimation_true_no_h.png for GPS_SETTING: {GPS_SETTING}")





















'''import pandas as pd
import numpy as np
# from models.lstm_model import LSTMModel # Removed as per request
from ekf_core import EKFCore
from units.quaternion import quaternion_to_euler
import os
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence # Removed as per request
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from TCN_csv.simpleTCN import SimpleTCN
from sklearn.preprocessing import StandardScaler
from TCN_csv.train_TCN_4 import FlightDataset, load_flight_data  # 从训练脚本中导入 FlightDataset
#from lstm_csv.test_lstm import preprocess


# ========== 数据加载与标准化 ==========
def preprocess(flight_ids, scaler_x, scaler_y):
    X_raw, Y_raw = load_flight_data(flight_ids)

    # 使用训练阶段的标准化参数
    X_scaled = scaler_x.transform(X_raw.reshape(-1, X_raw.shape[2])).reshape(X_raw.shape)
    Y_scaled = scaler_y.transform(Y_raw.reshape(-1, 1)).reshape(-1)

    return X_scaled, Y_scaled


# === Constants ===
g = 32.17  # 重力常数 (ft/s^2)
dt = 1/40  # 采样间隔
input_type = '4aixs' #'allinput','4axis'


# === User-defined settings ===
GPS_SETTING = 'false'  # 'false' or 'true'
OBJECT_SETTING = 'alpha'    # 'alpha' as per request

# === Load CSV Data ===
flight_id = 12
Flight_path = "Flight_" +str(flight_id) +".csv"
csv_path = os.path.join("../data/csv_segments", Flight_path)


df = pd.read_csv(csv_path, on_bad_lines='skip', sep=',') # Explicitly set delimiter

# Convert all relevant columns to numeric, coercing errors to NaN
for col in ['AOA', 'Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator', 'Pitch_rate', 'Roll_rate', 'Rudder', 'Vd', 'Ve', 'Vn', 'Yaw_rate', 'q0', 'q1', 'q2', 'q3']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any NaN values that resulted from coercion (malformed data)
df.dropna(inplace=True)

data = df.values

# Extract quaternion and convert to Euler angles
q = df[["q0", "q1", "q2", "q3"]].values
phi, theta, psi = quaternion_to_euler(q)
psi[psi < 0] += 2 * np.pi  # Map to [0, 2π]

# === Construct input sequences (um) ===
# Ensure all data used for um is converted to float
aileron = df['Aileron'].values.astype(np.float64)
elevator = df['Elevator'].values.astype(np.float64)
rudder = df['Rudder'].values.astype(np.float64)
airspeed = (df['Airspeed'].values * 3.2808).astype(np.float64)  # m/s -> ft/s
ax = (df['Acc_x'].values * 3.2808).astype(np.float64)
ay = (df['Acc_y'].values * 3.2808).astype(np.float64)
az = (df['Acc_z'].values * 3.2808).astype(np.float64)
p = df['Roll_rate'].values.astype(np.float64)
q_body = df['Pitch_rate'].values.astype(np.float64) # Renamed to avoid conflict with quaternion q
r = df['Yaw_rate'].values.astype(np.float64)

# === Initialize EKF ===
# Initial state vector x_init depends on GPS_SETTING
# States: [vt, alpha, beta, h] for false_no_h
# States: [vt, alpha, beta, h, wn, we, wd] for true_no_h

# Initial values from the first data point
vt_init = float(airspeed[0])
alpha_init = float(df['AOA'][0]) # Use initial AOA from data as initial guess
beta_init = 0.0 # Assume initial beta is zero

if GPS_SETTING == 'false':
    x0 = np.array([vt_init, alpha_init, beta_init], dtype=np.float64)
elif GPS_SETTING == 'true':
    wn_init = 0.0 # Initial wind biases
    we_init = 0.0
    wd_init = 0.0
    x0 = np.array([vt_init, alpha_init, beta_init, wn_init, we_init, wd_init], dtype=np.float64)
else:
    raise ValueError("Invalid GPS_SETTING.")

ekf = EKFCore(GPS_setting=GPS_SETTING, object_setting=OBJECT_SETTING, x_init=x0)


# 选择设备 (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === 初始化 LSTM 模型 ===
if input_type == 'allinput':
    input_dim = 10
elif input_type == '4aixs':
    input_dim = 4

TCN_model = SimpleTCN(input_dim=4,
                     channels=[10, 16, 32, 64],  # 4层，每层64个输出通道
                     kernel_size=3,
                     dropout=0.2).to(device)
TCN_model.load_state_dict(torch.load("../TCN_csv/tcn_model_4.pt", map_location=device))
TCN_model.eval()


def predictions(flight_id, scaler_x, scaler_y):
    #for flight_id in flight_ids:
    X_test, Y_test = preprocess([flight_id], scaler_x, scaler_y)  # 使用修正后的 preprocess
    test_loader = DataLoader(FlightDataset(X_test, Y_test), batch_size=1, shuffle=False)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_true.extend(Y_batch.numpy())

            pred = TCN_model(X_batch).cpu().numpy()
            y_pred.extend(pred)

    # 反标准化
    y_true = scaler_y.inverse_transform(np.array(y_true).reshape(-1, 1)).reshape(-1)
    y_pred = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).reshape(-1)

    return y_pred


# === Load LSTM Model for AOA Prediction ===
# Load normalization parameters
#norms = np.load("E:/tiaozhanbei/code/python/uav_angle_estimator/models/input_normalization_params_AOA.npz")
#mu = norms['mu']      # shape: (10,)
#sigma = norms['sigma']

scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaler_x.mean_ = np.load("../tcn_csv/input_mean_4.npy")
scaler_x.scale_ = np.load("../tcn_csv/input_std_4.npy")
scaler_y.mean_ = np.load("../tcn_csv/aoa_mean_4.npy")
scaler_y.scale_ = np.load("../tcn_csv/aoa_std_4.npy")

aoa_pre = predictions(flight_id, scaler_x, scaler_y)


# === Start Filtering ===
N = len(airspeed)
estimates = []
M = len(aoa_pre)

#time = min(N,M)

aoa_pre_new = np.zeros(N)

if N>M:
    for i in range(M):
        aoa_pre_new[i] = aoa_pre[i]
else:
    for i in range(N):
        aoa_pre_new[i] = aoa_pre[i]

aoa_true = df['AOA'].values.astype(np.float64)

# plt.ion()  # 开启交互模式
# fig, plot_ax = plt.subplots()  # ✅ 避免与 ax[t] 冲突
# line1, = plot_ax.plot([], [], 'b-', label='Measured AOA')
# line2, = plot_ax.plot([], [], 'r--', label='EKF Estimated AOA')
# plot_ax.set_xlim(0, N)
# plot_ax.set_ylim(min(aoa_true)-0.1, max(aoa_true)+0.1)
# plot_ax.set_xlabel("Time Step")
# plot_ax.set_ylabel("AOA (rad)")
# plot_ax.legend()
# plot_ax.grid(True)

# 存储历史数据
aoa_est_list = []
aoa_true_list = []


# for t in range(N):
#     # Construct EKF input (um)
#     # um: [p, q, r, ax, ay, az, phi, theta, psi]
#     um = np.array([p[t], q_body[t], r[t], ax[t], ay[t], az[t], phi[t], theta[t], psi[t]], dtype=np.float64)
#
#     # State prediction
#     ekf.predict(um, dt, g)
#
#     # 构造 LSTM 输入特征 xt
#     xt = np.array([
#         df['Acc_x'].iloc[t],
#         df['Acc_y'].iloc[t],
#         df['Acc_z'].iloc[t],
#         df['Aileron'].iloc[t],
#         df['Airspeed'].iloc[t],
#         df['Elevator'].iloc[t],
#         df['Pitch_rate'].iloc[t],
#         df['Roll_rate'].iloc[t],
#         df['Rudder'].iloc[t],
#         df['Yaw_rate'].iloc[t],
#     ], dtype=np.float32)
#
#
#     aoa_pred_re = torch.tensor(aoa_pre_new[t])
#
#     if GPS_SETTING == 'false':
#         # z = [airspeed[t], h_measurement, AOA[t]]
#         # Since h is not in CSV, we use 0.0 for h_measurement
#         z = np.array([airspeed[t], float(aoa_pred_re)], dtype=np.float64)
#     elif GPS_SETTING == 'true':
#         vn = df['Vn'].values.astype(np.float64)[t]
#         ve = df['Ve'].values.astype(np.float64)[t]
#         vd = df['Vd'].values.astype(np.float64)[t]
#         #z = np.array([airspeed[t], vn, ve, vd, aoa_pred_re], dtype=np.float64)
#         z = np.array([airspeed[t], vn, ve, vd, aoa_pred_re.cpu().numpy()], dtype=np.float64)
#     else:
#         raise ValueError("Invalid GPS_SETTING.")
#
#     # Update step
#     #print(torch.tensor(z).shape)
#     ekf.update(z, um)
#     estimates.append(ekf.get_state().copy())

    # === 实时绘图逻辑 ===
    # aoa_est_list.append(estimates[-1][1])
    # aoa_true_list.append(aoa_true[t])
    #
    # line1.set_data(range(len(aoa_true_list)), aoa_true_list)
    # line2.set_data(range(len(aoa_est_list)), aoa_est_list)
    #
    # plot_ax.set_xlim(0, max(100, t))  # ✅ 用 plot_ax 而不是 ax
    # plt.pause(0.01)

# plt.ioff()
# plt.show()

WINDOW_SIZE = 100
xt_history = []
estimates = []

for t in range(N):
    # ==== EKF 输入 um ====
    um = np.array([p[t], q_body[t], r[t], ax[t], ay[t], az[t], phi[t], theta[t], psi[t]], dtype=np.float64)
    ekf.predict(um, dt, g)

    if input_type == 'allinput':
        # ==== 构造 xt (当前时刻10维输入) ====
        xt = np.array([
            df['Acc_x'].iloc[t],
            df['Acc_y'].iloc[t],
            df['Acc_z'].iloc[t],
            df['Aileron'].iloc[t],
            df['Airspeed'].iloc[t],
            df['Elevator'].iloc[t],
            df['Pitch_rate'].iloc[t],
            df['Roll_rate'].iloc[t],
            df['Rudder'].iloc[t],
            df['Yaw_rate'].iloc[t],
        ], dtype=np.float32)

    elif input_type == '4aixs':
        # ==== 构造 xt (当前时刻4维输入) ====
        xt = np.array([
            df['Acc_z'].iloc[t],
            df['Airspeed'].iloc[t],
            df['Elevator'].iloc[t],
            df['Pitch_rate'].iloc[t],
        ], dtype=np.float32)


    # ==== 标准化并放入历史序列 ====
    xt_scaled = scaler_x.transform(xt.reshape(1, -1)).squeeze(0)
    xt_history.append(xt_scaled)

    # 不足窗口长度时跳过
    if len(xt_history) < WINDOW_SIZE:
        continue

    # ==== 构造LSTM输入 ====
    input_seq = np.array(xt_history[-WINDOW_SIZE:])  # shape: (100, 10)
    input_seq = np.expand_dims(input_seq, axis=0)    # shape: (1, 100, 10)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)

    # ==== LSTM 推理 ====
    with torch.no_grad():
        aoa_pred_norm = TCN_model(input_tensor).cpu().numpy().item()
    aoa_pred = scaler_y.inverse_transform([[aoa_pred_norm]])[0][0]

    # ==== 构造 EKF 观测向量 z ====
    if GPS_SETTING == 'false':
        z = np.array([airspeed[t], aoa_pred], dtype=np.float64)
    elif GPS_SETTING == 'true':
        vn = df['Vn'].values.astype(np.float64)[t]
        ve = df['Ve'].values.astype(np.float64)[t]
        vd = df['Vd'].values.astype(np.float64)[t]
        z = np.array([airspeed[t], vn, ve, vd, aoa_pred], dtype=np.float64)
    else:
        raise ValueError("Invalid GPS_SETTING.")

    # ==== EKF 更新 ====
    ekf.update(z, um)
    estimates.append(ekf.get_state().copy())



estimates = np.array(estimates)
valid_aoa_true = aoa_true[WINDOW_SIZE-1:]  # 裁剪掉前100个点




def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差(MAE)

    参数:
    y_true - 真实值的数组
    y_pred - 预测值的数组

    返回:
    mae - 平均绝对误差
    """
    # 确保输入为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算绝对值误差并取平均
    absolute_errors = np.abs(y_true - y_pred)
    mae = np.mean(absolute_errors)

    return mae

mae = calculate_mae(valid_aoa_true, estimates[:, 1])
print(f"AOA MAE: {mae:.4f}(rad)")


# === Visualization ===
plt.figure(figsize=(12, 6))
plt.plot(valid_aoa_true, label='Measured AOA (rad)', color='blue', alpha=0.7)
plt.plot(estimates[:, 1], label='EKF Estimated AOA (rad)', color='red', linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("Angle of Attack (rad)")
plt.legend()
plt.grid(True)
plt.title(f"EKF Real-Time AOA Estimation (GPS_SETTING: {GPS_SETTING})")
plt.tight_layout()
# plt.savefig('aoa_estimation_true_no_h.png')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(aoa_true, label='Measured AOA (rad)', color='blue', alpha=0.7)
plt.plot(aoa_pre, label='TCN Estimated AOA (rad)', color='red', linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("Angle of Attack (rad)")
plt.legend()
plt.grid(True)
plt.title(f"TCN Real-Time AOA Predict (GPS_SETTING: {GPS_SETTING})")
plt.tight_layout()
# plt.savefig('aoa_estimation_true_no_h.png')
plt.show()

print(f"AOA estimation plot saved to aoa_estimation_true_no_h.png for GPS_SETTING: {GPS_SETTING}")'''


















