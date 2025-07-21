import h5py
import numpy as np
import os
import pandas as pd

# === 配置路径 ===
h5_path = 'D:/code/uav_angle_estimator/data/raw/FD_cal.h5'
csv_dir = 'D:/code/uav_angle_estimator/data/csv_segments'
os.makedirs(csv_dir, exist_ok=True)

# === 索引映射（Matlab索引 - 1） ===
index_map = {
    'tas': 51,
    'aoa': 44,
    'roll_rate': 353,
    'pitch_rate': 347,
    'yaw_rate': 354,
    'elevator': 40,
    'left_aileron': 23,
    'right_aileron': 42,
    'rudder': 41,
    'acc_x': 272,
    'acc_y': 275,
    'acc_z': 278,
    'vel_n': 434,
    'vel_e': 432,
    'vel_d': 433,
    'q0': 348,
    'q1': 349,
    'q2': 350,
    'q3': 351
}

# === 加载 h5 文件 ===
with h5py.File(h5_path, 'r') as f:
    root = f['/']
    group_names = list(root.keys())  # ['/07_54_01.ulg', ...]

    for i, group_name in enumerate(group_names):
        group = f[group_name]
        dataset_names = list(group.keys())  # ['Dataset_0', ..., 'Dataset_547']
        # print(dataset_names)
        datasets = [group[k] for k in dataset_names]  # list of 548 datasets
        print(datasets[1])


