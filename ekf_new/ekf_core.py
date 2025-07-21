import numpy as np
from ekf_predict import state_eq, linearization
from ekf_update import ob_eq, linear_mea


class EKFCore:
    def __init__(self, GPS_setting='true', object_setting='alpha', x_init=None):
        self.GPS_setting = GPS_setting
        self.object_setting = object_setting

        if self.GPS_setting == 'false':
            self.n_state = 3  # vt, alpha, beta
            # R: [vt, h, alpha] measurements
            # From set_parameter.m: R = diag([1E-1 1E-1 1E-6]) for object='alpha', GPS='false'
            self.R = np.diag([1e-1, 1e-6])
            # Qx: [vt, alpha, beta] process noise
            # From set_parameter.m: Qx = diag([1E-2 1E-9 1E-9 1E-1]) for object='alpha', GPS='false'
            self.Q = np.diag([1e-2, 1e-9, 1e-9])
        elif self.GPS_setting == 'true':
            self.n_state = 6
            self.R = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-6])
            self.Q = np.diag([1e-3, 1e-4, 1e-9, 1e-9, 1e-6, 1e-15])
        else:
            raise ValueError("Invalid GPS_setting. Must be 'false_no_h' or 'true_no_h'.")

        # Set P0: Initial covariance
        if x_init is None:
            if self.GPS_setting == 'false':
                x_init = np.zeros(self.n_state)
            elif self.GPS_setting == 'true':
                x_init = np.zeros(self.n_state)
        self.x_est = x_init.copy()

        if self.GPS_setting == 'false':
            w = np.array([0.05 * float(x_init[0]), np.pi / 180, np.pi / 180])
        elif self.GPS_setting == 'true':
            w = np.array([0.05 * float(x_init[0]), np.pi / 180, np.pi / 180, 0.05 * float(x_init[3]), 0.05 * float(x_init[4]), 0.0005])
        self.P = np.diag(w**2)

    def reset(self, x0=None, P0=None):
        if x0 is not None:
            self.x_est = x0.copy()
        if P0 is not None:
            self.P = P0.copy()

    def predict(self, um, dt, g):
        gps_flag_for_func = self.GPS_setting

        # State prediction
        self.x_est = state_eq(self.x_est, um, g, dt, gps_flag_for_func)
        F = linearization(self.x_est, um, g, dt, self.n_state, gps_flag_for_func)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, um):
        gps_flag_for_func = self.GPS_setting

        H = linear_mea(self.x_est, um, self.object_setting, gps_flag_for_func)
        h_xpre = ob_eq(self.x_est, um, self.object_setting, gps_flag_for_func)

        y = z - h_xpre.flatten()
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x_est = self.x_est + (K @ y).flatten()
        self.P = (np.eye(self.n_state) - K @ H) @ self.P

    def get_state(self):
        return self.x_est

    def get_covariance(self):
        return self.P


