import numpy as np

#运用欧拉法得到x_predicate 其中statse、um均为向量
def state_eq(states, um, g, dt, GPS):
    if GPS == 'false':
        vt = states[0]
        alpha = states[1]
        beta = states[2]
    elif GPS == 'true':
        vt = states[0]
        alpha = states[1]
        beta = states[2]
        w_n = states[3]
        w_e = states[4]
        w_d = states[5]

    # Adjusted indices for um
    p = um[0]
    q = um[1]
    r = um[2]
    ax = um[3]
    ay = um[4]
    az = um[5]
    phi = um[6]
    theta = um[7]
    psi = um[8]

    saf = np.sin(alpha)
    caf = np.cos(alpha)
    sbta = np.sin(beta)
    cbta = np.cos(beta)
    tbta = np.tan(beta)
    sth = np.sin(theta)
    cth = np.cos(theta)
    sph = np.sin(phi)
    cph = np.cos(phi)

    if GPS == 'false':
        xdot = np.array([
            g * (-sth * caf * cbta + sph * cth * sbta + cph * cth * saf * cbta) + ax * caf * cbta + ay * sbta + az * saf * cbta,
            (g * (cph * cth * caf + sth * saf) + az * caf - ax * saf) / (vt * cbta) + q - tbta * (p * caf + r * saf),
            (g * (sth * caf * sbta + sph * cth * cbta - cph * cth * saf * sbta) - ax * caf * sbta + ay * cbta - az * saf * sbta) / vt + p * saf - r * caf
        ])
        X_predicate = states + dt * xdot

    elif GPS == 'true':
        w_n_noise = 0.1 * np.random.randn(1)[0]
        w_e_noise = 0.1 * np.random.randn(1)[0]
        w_d_noise = 0.00005 * np.random.randn(1)[0]

        xdot = np.array([
            g * (-sth * caf * cbta + sph * cth * sbta + cph * cth * saf * cbta) + ax * caf * cbta + ay * sbta + az * saf * cbta,
            (g * (cph * cth * caf + sth * saf) + az * caf - ax * saf) / (vt * cbta) + q - tbta * (p * caf + r * saf),
            (g * (sth * caf * sbta + sph * cth * cbta - cph * cth * saf * sbta) - ax * caf * sbta + ay * cbta - az * saf * sbta) / vt + p * saf - r * caf,
            w_n_noise,
            w_e_noise,
            w_d_noise
        ])
        X_predicate = states + dt * xdot

    return X_predicate

def linearization(x_est, um, g, t_delta, n_state, GPS):

    # 状态变量
    if GPS == 'false':
        vt, alpha, beta = x_est[:3]
    elif GPS == 'true':
        vt, alpha, beta, w_n, w_e, w_d = x_est[:6]

    # 控制输入 (Adjusted indices for um)
    p = um[0]
    q = um[1]
    r = um[2]
    ax = um[3]
    ay = um[4]
    az = um[5]
    phi = um[6]
    theta = um[7]
    psi = um[8]

    # 计算三角函数
    saf, caf = np.sin(alpha), np.cos(alpha)
    sbta, cbta = np.sin(beta), np.cos(beta)
    tbta, sebta = np.tan(beta), 1 / np.cos(beta)
    sth, cth = np.sin(theta), np.cos(theta)
    sph, cph = np.sin(phi), np.cos(phi)

    if GPS == 'false':
        A = np.array([
            [0, g * t_delta * (cbta * saf * sth + caf * cbta * cph * cth) + az * t_delta * caf * cbta - ax * t_delta * cbta * saf,
             g * t_delta * (cbta * cth * sph + caf * sbta * sth - cph * saf * sbta * cth) + ay * t_delta * cbta - ax * t_delta * caf * sbta - az * t_delta * saf * sbta],
            [(ax * t_delta * saf) / (vt**2 * cbta) - (g * t_delta * (saf * sth + caf * cph * cth)) / (vt**2 * cbta) - (az * t_delta * caf) / (vt**2 * cbta),
             p * t_delta * saf * tbta - r * t_delta * caf * tbta - (az * t_delta * saf) / (vt * cbta) + (g * t_delta * (caf * sth - cph * saf * cth)) / (vt * cbta) - (ax * t_delta * caf) / (vt * cbta),
             (g * t_delta * sbta * (saf * sth + caf * cph * cth)) / (vt * cbta**2) - r * t_delta * saf * (tbta**2 + 1) - p * t_delta * caf * (tbta**2 + 1) +
             (az * t_delta * caf * sbta) / (vt * cbta**2) - (ax * t_delta * saf * sbta) / (vt * cbta**2)],
            [(ax * t_delta * caf * sbta) / vt**2 - (g * t_delta * (cbta * cth * sph + caf * sbta * sth - cph * saf * sbta * cth)) / vt**2 - (ay * t_delta * cbta) / vt**2 +
             (az * t_delta * saf * sbta) / vt**2,
             p * t_delta * caf + r * t_delta * saf - (g * t_delta * (saf * sbta * sth + caf * cph * sbta * cth)) / vt - (az * t_delta * caf * sbta) / vt + (ax * t_delta * saf * sbta) / vt,
             - (ay * t_delta * sbta) / vt - (ax * t_delta * caf * cbta) / vt - (az * t_delta * cbta * saf) / vt - (g * t_delta * (sbta * cth * sph - caf * cbta * sth + cbta * cph * saf * cth)) / vt]
        ])
        F = A + np.eye(n_state)

    elif GPS == 'true':
        A = np.array([
            [0, g * t_delta * (cbta * saf * sth + caf * cbta * cph * cth) + az * t_delta * caf * cbta - ax * t_delta * cbta * saf,
             g * t_delta * (cbta * cth * sph + caf * sbta * sth - cph * saf * sbta * cth) + ay * t_delta * cbta - ax * t_delta * caf * sbta - az * t_delta * saf * sbta,
             0, 0, 0],

            [(ax * t_delta * saf) / (vt**2 * cbta) - (g * t_delta * (saf * sth + caf * cph * cth)) / (vt**2 * cbta) - (az * t_delta * caf) / (vt**2 * cbta),
             p * t_delta * saf * tbta - r * t_delta * caf * tbta - (az * t_delta * saf) / (vt * cbta) + (g * t_delta * (caf * sth - cph * saf * cth)) / (vt * cbta) - (ax * t_delta * caf) / (vt * cbta),
             (g * t_delta * sbta * (saf * sth + caf * cph * cth)) / (vt * cbta**2) - r * t_delta * saf * (tbta**2 + 1) - p * t_delta * caf * (tbta**2 + 1) +
             (az * t_delta * caf * sbta) / (vt * cbta**2) - (ax * t_delta * saf * sbta) / (vt * cbta**2),
             0, 0, 0],

            [(ax * t_delta * caf * sbta) / vt**2 - (g * t_delta * (cbta * cth * sph + caf * sbta * sth - cph * saf * sbta * cth)) / vt**2 - (ay * t_delta * cbta) / vt**2 +
             (az * t_delta * saf * sbta) / vt**2,
             p * t_delta * caf + r * t_delta * saf - (g * t_delta * (saf * sbta * sth + caf * cph * sbta * cth)) / vt - (az * t_delta * caf * sbta) / vt + (ax * t_delta * saf * sbta) / vt,
             - (ay * t_delta * sbta) / vt - (ax * t_delta * caf * cbta) / vt - (az * t_delta * cbta * saf) / vt - (g * t_delta * (sbta * cth * sph - caf * cbta * sth + cbta * cph * saf * cth)) / vt,
             0, 0, 0],

            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        F = A + np.eye(n_state)

    return F

