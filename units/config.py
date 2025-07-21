# utils/config.py

from models.ETH_lstm_4.model import LSTMModel as ETH_LSTM
from models.ETH_tcn_4.simpleTCN import SimpleTCN as ETH_TCN
from models.MY_lstm_10.model import LSTMModel as MY_LSTM
from models.MY_tcn_4.simpleTCN import SimpleTCN as MY_TCN

MODEL_CONFIGS = {
    'eth_lstm_4': {
        'input_dim': 4,
        'features': ['Acc_z', 'Airspeed', 'Elevator', 'Pitch_rate'],
        'model_path': '../models/ETH_lstm_4/lstm_model_4.pt',
        'scaler_path': {
            'x': '../models/ETH_lstm_4/input_mean_4.npy',
            'x_std': '../models/ETH_lstm_4/input_std_4.npy',
            'y': '../models/ETH_lstm_4/aoa_mean_4.npy',
            'y_std': '../models/ETH_lstm_4/aoa_std_4.npy'
        },
        'model_class': ETH_LSTM
    },

    'eth_lstm_10': {
        'input_dim': 10,
        'features': ['Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed',
                     'Elevator', 'Pitch_rate', 'Roll_rate', 'Rudder', 'Yaw_rate'],
        'model_path': '../models/ETH_lstm_10/lstm_model.pt',
        'scaler_path': {
            'x': '../models/ETH_lstm_10/input_mean.npy',
            'x_std': '../models/ETH_lstm_10/input_std.npy',
            'y': '../models/ETH_lstm_10/aoa_mean.npy',
            'y_std': '../models/ETH_lstm_10/aoa_std.npy'
        },
        'model_class': ETH_LSTM
    },

    'eth_tcn_4': {
        'input_dim': 4,
        'features': ['Acc_z', 'Airspeed', 'Elevator', 'Pitch_rate'],
        'model_path': '../models/ETH_tcn_4/tcn_model_matlab_aligned.pt',
        'scaler_path': {
            'x': '../models/ETH_tcn_4/scaler_x.pkl',
            'y': '../models/ETH_tcn_4/scaler_y.pkl'
        },
        'model_class': ETH_TCN
    },

    'my_lstm_10': {
        'input_dim': 10,
        'features': ['Acc_x', 'Acc_y', 'Acc_z', 'Aileron', 'Airspeed',
                     'Elevator', 'Pitch_rate', 'Roll_rate', 'Rudder', 'Yaw_rate'],
        'model_path': '../models/MY_lstm_10/lstm_model.pt',
        'scaler_path': {
            'x': '../models/MY_lstm_10/my_input_mean.npy',
            'x_std': '../models/MY_lstm_10/my_input_std.npy',
            'y': '../models/MY_lstm_10/my_aoa_mean.npy',
            'y_std': '../models/MY_lstm_10/my_aoa_std.npy'
        },
        'model_class': MY_LSTM
    },

    'my_tcn_4': {
        'input_dim': 4,
        'features': ['Acc_z', 'Airspeed', 'Elevator', 'Pitch_rate'],
        'model_path': '../models/MY_tcn_4/my_tcn_model.pt',
        'scaler_path': {
            'x': '../models/MY_tcn_4/scaler_x.pkl',
            'y': '../models/MY_tcn_4/scaler_y.pkl'
        },
        'model_class': MY_TCN
    }
}
