import numpy as np

def quaternion_to_euler(q):
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 ** 2 + q2 ** 2)
    phi = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q0 * q2 - q3 * q1)
    theta = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 ** 2 + q3 ** 2)
    psi = np.arctan2(siny_cosp, cosy_cosp)

    return phi, theta, psi
