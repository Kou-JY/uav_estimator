# utils/model_loader.py

import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from units.config import MODEL_CONFIGS


def load_model_and_scaler(data_source: str, model_type: str):
    """
    根据 data_source 和 model_type 加载模型、标准化器和特征配置。

    Returns:
        model: 加载好权重的 PyTorch 模型（自动 .eval()）
        scaler_x: 输入标准化器（StandardScaler or joblib）
        scaler_y: 输出标准化器（StandardScaler or joblib）
        config: 当前模型配置项（含 features、input_dim 等）
    """
    key = f"{data_source}_{model_type}"
    config = MODEL_CONFIGS[key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 模型构造与加载 ===
    model = config['model_class'](input_dim=config['input_dim']).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()

    # === 标准化器加载 ===
    if 'x_std' in config['scaler_path']:  # LSTM 使用 .npy 格式
        scaler_x = StandardScaler()
        scaler_x.mean_ = np.load(config['scaler_path']['x'])
        scaler_x.scale_ = np.load(config['scaler_path']['x_std'])

        scaler_y = StandardScaler()
        scaler_y.mean_ = np.load(config['scaler_path']['y'])
        scaler_y.scale_ = np.load(config['scaler_path']['y_std'])
    else:  # TCN 使用 joblib 序列化
        scaler_x = joblib.load(config['scaler_path']['x'])
        scaler_y = joblib.load(config['scaler_path']['y'])

    return model, scaler_x, scaler_y, config
