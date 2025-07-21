import os
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ekf_new.ekf_core import EKFCore
from units.quaternion import quaternion_to_euler
from lstm_csv.train_lstm import FlightDataset, load_flight_data

# ===== 模型加载函数 =====
def load_model_and_scaler(model_type, input_dim):
    if model_type == 'lstm_4':
        from models.ETH_lstm_4.model import LSTMModel
        model_path = "models/ETH_lstm_4/lstm_model_4.pt"
        x_mean = np.load("models/ETH_lstm_4/input_mean_4.npy")
        x_std = np.load("models/ETH_lstm_4/input_std_4.npy")
        y_mean = np.load("models/ETH_lstm_4/aoa_mean_4.npy")
        y_std = np.load("models/ETH_lstm_4/aoa_std_4.npy")

    elif model_type == 'lstm_10':
        from models.ETH_lstm_10.model import LSTMModel
        model_path = "models/ETH_lstm_10/lstm_model.pt"
        x_mean = np.load("models/ETH_lstm_10/input_mean.npy")
        x_std = np.load("models/ETH_lstm_10/input_std.npy")
        y_mean = np.load("models/ETH_lstm_10/aoa_mean.npy")
        y_std = np.load("models/ETH_lstm_10/aoa_std.npy")

    elif model_type == 'tcn_4':
        from models.ETH_tcn_4.simpleTCN import TCNModel as LSTMModel
        model_path = "models/ETH_tcn_4/tcn_model_matlab_aligned.pt"
        with open("models/ETH_tcn_4/scaler_x.pkl", 'rb') as f:
            x_scaler = pickle.load(f)
        with open("models/ETH_tcn_4/scaler_y.pkl", 'rb') as f:
            y_scaler = pickle.load(f)
    else:
        raise ValueError("Unsupported model_type")

    model = LSTMModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if model_type.startswith("tcn"):
        return model, x_scaler, y_scaler
    else:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_x.mean_, scaler_x.scale_ = x_mean, x_std
        scaler_y.mean_, scaler_y.scale_ = y_mean, y_std
        return model, scaler_x, scaler_y

# ===== 参数设定 =====
flight_id = 12
model_type = 'lstm_10'  # 'lstm_4', 'lstm_10', or 'tcn_4'
GPS_SETTING = 'false'
OBJECT_SETTING = 'alpha'
dt = 1 / 40
g = 32.17
WINDOW_SIZE = 100

# ===== 初始化 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dims = {'lstm_4': 4, 'lstm_10': 10, 'tcn_4': 4}
model, scaler_x, scaler_y = load_model_and_scaler(model_type, input_dims[model_type])

# ===== 数据加载 =====
csv_path = os.path.join("data", "csv_segments", f"Flight_{flight_id}.csv")
df = pd.read_csv(csv_path)

for col in ['AOA', 'Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator',
            'Pitch_rate', 'Roll_rate', 'Rudder', 'Vd', 'Ve', 'Vn', 'Yaw_rate', 'q0', 'q1', 'q2', 'q3']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

q = df[['q0', 'q1', 'q2', 'q3']].values
phi, theta, psi = quaternion_to_euler(q)
psi[psi < 0] += 2 * np.pi

ax = (df['Acc_x'].values * 3.2808).astype(np.float64)
ay = (df['Acc_y'].values * 3.2808).astype(np.float64)
az = (df['Acc_z'].values * 3.2808).astype(np.float64)
airspeed = (df['Airspeed'].values * 3.2808).astype(np.float64)
p = df['Roll_rate'].values.astype(np.float64)
q_body = df['Pitch_rate'].values.astype(np.float64)
r = df['Yaw_rate'].values.astype(np.float64)

vt_init = float(airspeed[0])
alpha_init = float(df['AOA'].iloc[0])
beta_init = 0.0
x0 = np.array([vt_init, alpha_init, beta_init], dtype=np.float64)
ekf = EKFCore(GPS_setting=GPS_SETTING, object_setting=OBJECT_SETTING, x_init=x0)

xt_history = []
estimates = []
N = len(df)

# ===== 主循环 =====
for t in range(N):
    um = np.array([p[t], q_body[t], r[t], ax[t], ay[t], az[t], phi[t], theta[t], psi[t]], dtype=np.float64)
    ekf.predict(um, dt, g)

    if model_type == 'lstm_4' or model_type == 'tcn_4':
        xt = df[['Airspeed', 'Elevator', 'Pitch_rate', 'Acc_z']].iloc[t].values.astype(np.float32)
    else:
        xt = df[['Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator', 'Pitch_rate', 'Roll_rate', 'Rudder', 'Yaw_rate']].iloc[t].values.astype(np.float32)

    xt_scaled = scaler_x.transform(xt.reshape(1, -1)).squeeze(0)
    xt_history.append(xt_scaled)

    if len(xt_history) < WINDOW_SIZE:
        continue

    input_seq = np.array(xt_history[-WINDOW_SIZE:])
    input_seq = np.expand_dims(input_seq, axis=0)  # (1, 100, dim)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        aoa_pred_norm = model(input_tensor).cpu().numpy().item()
    aoa_pred = scaler_y.inverse_transform([[aoa_pred_norm]])[0][0]

    if GPS_SETTING == 'false':
        z = np.array([airspeed[t], aoa_pred], dtype=np.float64)
    elif GPS_SETTING == 'true':
        vn, ve, vd = df['Vn'].iloc[t], df['Ve'].iloc[t], df['Vd'].iloc[t]
        z = np.array([airspeed[t], vn, ve, vd, aoa_pred], dtype=np.float64)

    ekf.update(z, um)
    estimates.append(ekf.get_state().copy())

# ===== 可视化 =====
import matplotlib.pyplot as plt
estimates = np.array(estimates)
aoa_true = df['AOA'].values.astype(np.float64)[-len(estimates):]

plt.figure(figsize=(12, 6))
plt.plot(aoa_true, label='Measured AOA (rad)', color='blue', alpha=0.7)
plt.plot(estimates[:, 1], label='EKF Estimated AOA (rad)', color='red', linestyle='--')
plt.xlabel("Time Step")
plt.ylabel("Angle of Attack (rad)")
plt.legend()
plt.grid(True)
plt.title(f"EKF Real-Time AOA Estimation (GPS_SETTING: {GPS_SETTING}, Model: {model_type})")
plt.tight_layout()
plt.show()
