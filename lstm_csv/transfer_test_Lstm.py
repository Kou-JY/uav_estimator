import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from lstm_csv.model import LSTMModel

# ===== 配置参数 =====
DATA_PATH = "../data/my_csv_segments/Flight_my_03.csv"
MODEL_PATH = "../models/Transfer_lstm_10/transfer_lstm_model.pt"
SCALER_X_PATH = "../models/Transfer_lstm_10/transfer_input_scaler.pkl"
SCALER_Y_PATH = "../models/Transfer_lstm_10/transfer_aoa_scaler.pkl"
WINDOW_SIZE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r2d = 57.295779513  # 弧度转角度

# ===== 加载数据 =====
df = pd.read_csv(DATA_PATH)
df = df.dropna()

X_raw = df[['Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator',
            'Pitch_rate', 'Roll_rate', 'Rudder', 'Yaw_rate']].values
Y_true = df['AOA'].values.astype(np.float32)

# ===== 加载归一化器 =====
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

X_scaled = scaler_x.transform(X_raw)

# ===== 加载模型 =====
model = LSTMModel(input_dim=10).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== 滑动窗口预测 =====
preds = []
for i in range(WINDOW_SIZE, len(X_scaled)):
    input_seq = X_scaled[i-WINDOW_SIZE:i]  # shape (100, 10)
    input_tensor = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_norm = model(input_tensor).cpu().numpy().item()
        pred_aoa = scaler_y.inverse_transform([[pred_norm]])[0][0]
    preds.append(pred_aoa)

# ===== 裁剪 ground truth 与 RMSE 计算 =====
Y_true_cropped = Y_true[WINDOW_SIZE:]
preds = np.array(preds)
rmse = np.sqrt(np.mean((Y_true_cropped - preds)**2))
print(f"RMSE between predicted AOA and measured AOA: {rmse:.4f} rad ({r2d * rmse:.2f} deg)")

# ===== 可视化 =====
plt.figure(figsize=(12, 6))
plt.plot(Y_true_cropped, label="Measured AOA (rad)", color='blue', alpha=0.6)
plt.plot(preds, label="Predicted AOA (rad)", color='red', linestyle='--')
plt.title("AOA Prediction vs Measurement (Flight_my_03)")
plt.xlabel("Time Step")
plt.ylabel("Angle of Attack (rad)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
