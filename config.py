# config.py

MODEL_TYPE = 'lstm'       # 'lstm' or 'tcn'
INPUT_DIM = 4             # 4 or 10
PLATFORM = 'ETH'          # 'ETH' or 'MYUAV'

MODEL_PATHS = {
    ('ETH', 'lstm', 4): 'models/ETH_lstm_4/lstm_model_4.pt',
    ('ETH', 'lstm', 10): 'models/ETH_lstm_10/lstm_model.pt',
    ('ETH', 'tcn', 4): 'models/ETH_tcn_4/tcn_model_matlab_aligned.pt',
    ('MYUAV', 'lstm', 4): 'models/MYUAV_lstm_4/lstm_model.pt',
    ('MYUAV', 'tcn', 4): 'models/MYUAV_tcn_4/tcn_model.pt',
}

SCALER_PATHS = {
    ('ETH', 'lstm', 4): (
        'models/ETH_lstm_4/input_mean_4.npy',
        'models/ETH_lstm_4/input_std_4.npy',
        'models/ETH_lstm_4/aoa_mean_4.npy',
        'models/ETH_lstm_4/aoa_std_4.npy'
    ),
    ('ETH', 'lstm', 10): (
        'models/ETH_lstm_10/input_mean.npy',
        'models/ETH_lstm_10/input_std.npy',
        'models/ETH_lstm_10/aoa_mean.npy',
        'models/ETH_lstm_10/aoa_std.npy'
    ),
    ('ETH', 'tcn', 4): (
        'models/ETH_tcn_4/scaler_x.pkl',
        'models/ETH_tcn_4/scaler_y.pkl'
    ),
    ('MYUAV', 'lstm', 4): (
        'models/MYUAV_lstm_4/input_mean_4.npy',
        'models/MYUAV_lstm_4/input_std_4.npy',
        'models/MYUAV_lstm_4/aoa_mean_4.npy',
        'models/MYUAV_lstm_4/aoa_std_4.npy'
    ),
    ('MYUAV', 'tcn', 4): (
        'models/MYUAV_tcn_4/scaler_x.pkl',
        'models/MYUAV_tcn_4/scaler_y.pkl'
    ),
}
