# utils/data_loader.py

import os
import numpy as np
import pandas as pd
from units.quaternion import quaternion_to_euler


def load_flight_data_for_ekf(data_source: str, flight_id: int) -> dict:
    """
    加载飞行数据，返回适用于 EKF 的所有输入变量。
    支持 'eth'（开源数据，姿态由四元数计算）和 'my'（自研数据，姿态角已记录）两类数据。

    Returns:
        dict: 包含 ax, ay, az, airspeed, p, q, r, vn, ve, vd, phi, theta, psi, aoa 等字段
    """
    if data_source == 'eth':
        csv_path = os.path.join("../data", "csv_segments_new", f"Flight_{flight_id}.csv")
    elif data_source == 'my':
        csv_path = os.path.join("../data", "my_csv_segments", f"Flight_my_{flight_id:02d}.csv")
    else:
        raise ValueError("data_source must be 'eth' or 'my'")

    df = pd.read_csv(csv_path)

    # 统一解析所需字段
    base_cols = ['AOA', 'Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed', 'Elevator',
                 'Pitch_rate', 'Roll_rate', 'Rudder', 'Vd', 'Ve', 'Vn', 'Yaw_rate']
    for col in base_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if data_source == 'eth':
        for col in ['q0', 'q1', 'q2', 'q3']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        q = df[['q0', 'q1', 'q2', 'q3']].values
        phi, theta, psi = quaternion_to_euler(q)
        psi[psi < 0] += 2 * np.pi
    elif data_source == 'my':
        for col in ['Roll', 'Pitch', 'Yaw']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        phi = df['Roll'].values.astype(np.float64)
        theta = df['Pitch'].values.astype(np.float64)
        psi = df['Yaw'].values.astype(np.float64)

    # 转单位（默认 SI → 英尺/弧度等）
    to_ft = 3.2808
    ax = (df['Acc_x'].values * to_ft).astype(np.float64)
    ay = (df['Acc_y'].values * to_ft).astype(np.float64)
    az = (df['Acc_z'].values * to_ft).astype(np.float64)
    airspeed = (df['Airspeed'].values * to_ft).astype(np.float64)
    p = df['Roll_rate'].values.astype(np.float64)
    q_body = df['Pitch_rate'].values.astype(np.float64)
    r = df['Yaw_rate'].values.astype(np.float64)
    vn = (df['Vn'].values * to_ft).astype(np.float64)
    ve = (df['Ve'].values * to_ft).astype(np.float64)
    vd = (df['Vd'].values * -to_ft).astype(np.float64)  # 注意：Z向下

    aoa = df['AOA'].values.astype(np.float64)

    return {
        'aoa': aoa,
        'ax': ax, 'ay': ay, 'az': az,
        'airspeed': airspeed,
        'p': p, 'q': q_body, 'r': r,
        'vn': vn, 've': ve, 'vd': vd,
        'phi': phi, 'theta': theta, 'psi': psi,
        'raw_df': df
    }
