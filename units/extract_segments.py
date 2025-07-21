import h5py
import numpy as np
import os
import torch

# === 配置路径 ===

h5_path = 'D:/code/uav_angle_estimator/data/raw/FD_cal.h5'
save_dir = 'D:/code/uav_angle_estimator/data/processed'
os.makedirs(save_dir, exist_ok=True)
NEUTRAL_POSITION = 1500
SCALE_FACTOR = 500
MAX_ANGLE_RAD = 0.5236


# === 所需变量索引（MATLAB 索引 - 1） ===
# input_indices = [23, 40, 41, 42, 51, 272, 275, 278, 347, 353, 354]
control_indices = {
    'Left_Aileron': 23,   # 24-1
    'Elevator': 40,       # 41-1
    'Rudder': 41,         # 42-1
    'Right_Aileron': 42   # 43-1
}
other_input_indices = [51, 272, 275, 278, 347, 353, 354]
output_indices = {'AOA': 44, 'Slip': 46}

# === 读取并处理每段飞行数据 ===
with h5py.File(h5_path, 'r') as f:
    root = f['/']
    group_names = list(root.keys())  # ['/07_54_01.ulg', ...]

    for i, group_name in enumerate(group_names):
        try:
            group = f[group_name]
            dataset_names = list(group.keys())  # ['Dataset_0', ..., 'Dataset_547']
            datasets = [group[k] for k in dataset_names]  # list of 548 datasets

            # === 提取舵面控制信号并转换 ===
            # 获取原始PWM信号
            pwm_left_ail = datasets[control_indices['Left_Aileron']][()]
            pwm_right_ail = datasets[control_indices['Right_Aileron']][()]
            pwm_elevator = datasets[control_indices['Elevator']][()]
            pwm_rudder = datasets[control_indices['Rudder']][()]

            # 转换为弧度 (按照您提供的公式)
            left_ail_rad = ((pwm_left_ail - NEUTRAL_POSITION) / SCALE_FACTOR) * MAX_ANGLE_RAD
            right_ail_rad = ((pwm_right_ail - NEUTRAL_POSITION) / SCALE_FACTOR) * MAX_ANGLE_RAD
            elevator_rad = ((pwm_elevator - NEUTRAL_POSITION) / SCALE_FACTOR) * MAX_ANGLE_RAD
            rudder_rad = ((pwm_rudder - NEUTRAL_POSITION) / SCALE_FACTOR) * MAX_ANGLE_RAD

            # 计算平均副翼角
            aileron_rad = 0.5 * (left_ail_rad + right_ail_rad)

            # === 提取输入 ===
            other_inputs = [datasets[j][()] for j in other_input_indices]

            # === 组合所有输入 ===
            # 将转换后的舵面控制信号放在前面
            control_inputs = np.stack([
                elevator_rad,
                rudder_rad,
                aileron_rad  # 添加平均副翼角
            ], axis=1)

            # 组合其他输入
            other_inputs = np.stack(other_inputs, axis=1)

            # 合并所有输入特征
            X = np.concatenate([control_inputs, other_inputs], axis=1)

            # === 提取输出 ===
            aoa = datasets[output_indices['AOA']][()]
            slip = datasets[output_indices['Slip']][()]

            # === 去除无效尾部 ===
            L = min(len(aoa), len(slip), len(X)) - 1000
            X, aoa, slip = X[:L], aoa[:L], slip[:L]

            # === 保存为 PyTorch tensor ===
            data = {
                'Inputs': torch.tensor(X, dtype=torch.float32),
                'AOA': torch.tensor(aoa, dtype=torch.float32),
                'Slip': torch.tensor(slip, dtype=torch.float32)
            }
            out_name = f'segment_{i+1}.pt'
            torch.save(data, os.path.join(save_dir, out_name))
            print(f"✅ Saved {out_name} ({L} samples)")

        except Exception as e:
            print(f"❌ Failed on segment {i+1} ({group_name}): {e}")