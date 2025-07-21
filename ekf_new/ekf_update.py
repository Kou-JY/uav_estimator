import numpy as np

# Constants from MATLAB measure.m and ob_eq.m


def ob_eq(x_pre, um, object_setting, GPS_setting):
    # um: [p, q, r, ax, ay, az, phi, theta, psi]
    p = um[0]
    q = um[1]
    r = um[2]
    ax = um[3]
    ay = um[4]
    az = um[5]
    phi = um[6]
    theta = um[7]
    psi = um[8]

    # de, da, dr are not directly from um in the Python version,
    # they are control surface inputs in MATLAB um(1:3).
    # For ob_eq, they are used in aerodynamic coefficients calculation.
    # Since the CSV does not provide these, and the MATLAB ob_eq uses them as um(1), um(2), um(3),
    # we need to decide how to handle them. For now, let's assume they are zero or constant if not provided.
    # Based on the MATLAB `main.m` and `measure.m`, `de`, `da`, `dr` are from `u_km` which is the input.
    # In `ekf_lstm.py`, `um` is constructed without `aileron`, `elevator`, `rudder`.
    # Let's set them to 0 for now, as they are control inputs and might not be directly observable.
    de = 0.0 # Placeholder, not available in current um
    da = 0.0 # Placeholder, not available in current um
    dr = 0.0 # Placeholder, not available in current um

    vt = x_pre[0] # vt is always the first state variable
    alpha = x_pre[1]
    beta = x_pre[2]

    vn, ve, vd = 0.0, 0.0, 0.0 # Initialize vn, ve, vd for false_no_h case

    if GPS_setting == 'true': # Changed from 'true_no_h' to 'true' to match ekf_core.py
        wn = x_pre[3]
        we = x_pre[4]
        wd = x_pre[5]

        sa = np.sin(alpha)
        ca = np.cos(alpha)
        sb = np.sin(beta)
        cb = np.cos(beta)
        sth = np.sin(theta)
        cth = np.cos(theta)
        sph = np.sin(phi)
        cph = np.cos(phi)
        sps = np.sin(psi)
        cps = np.cos(psi)

        vn = vt * (ca * cb * cps * cth + sb * (-sps * cph + cps * sth * sph) + sa * cb * (sph * sps + cph * sth * cps)) + wn
        ve = vt * (ca * cb * cth * sps + sb * (sph * sth * sps + cph * cps) + sa * cb * (-sph * cps + cph * sth * sps)) + we
        vd = vt * (ca * cb * sth - sb * sph * cth - sa * cb * cph * cth) + wd

    # qbar is not passed, need to calculate it or assume it's part of um if it's a measurement
    # For now, let's assume qbar is not directly used in the observation equation for alpha estimation
    # The original MATLAB ob_eq uses qbar for aerodynamic coefficients, but those are pre-calculated constants here.
    # So, qbar is not needed for h_xpre calculation directly.


    # Ensure h_xpre is always assigned a value
    if object_setting == 'alpha':
        if GPS_setting == 'false': # Changed from 'false_no_h' to 'false'
            h_xpre = np.array([vt, alpha])
        elif GPS_setting == 'true': # Changed from 'true_no_h' to 'true'
            h_xpre = np.array([vt, vn, ve, vd, alpha])
        else:
            # This case should not be reached if GPS_setting is correctly handled in ekf_core.py
            # and ekf_lstm.py, but added for robustness.
            raise ValueError(f"Invalid GPS_setting: {GPS_setting} for object_setting=\'alpha\'.")
    else:
        # Handle other object_setting cases or raise an error if not supported
        # For now, we are only supporting 'alpha' estimation.
        raise ValueError("Unsupported object_setting. Only \'alpha\' is supported.")

    return h_xpre

def linear_mea(x_pre, um, object_setting, GPS_setting):
    # um: [p, q, r, ax, ay, az, phi, theta, psi]
    p = um[0]
    q = um[1]
    r = um[2]
    ax = um[3]
    ay = um[4]
    az = um[5]
    phi = um[6]
    theta = um[7]
    psi = um[8]

    # de, da, dr are not directly from um in the Python version,
    # they are control surface inputs in MATLAB um(1:3).
    # For linear_mea, they are used in Der_dal, Der_dbt, Der_dv which are not directly translated.
    # The linear_mea in Python directly computes H matrix based on state derivatives.
    # So, de, da, dr are not needed here.

    vt = x_pre[0]
    alpha = x_pre[1]
    beta = x_pre[2]

    sa = np.sin(alpha)
    ca = np.cos(alpha)
    sb = np.sin(beta)
    cb = np.cos(beta)
    sth = np.sin(theta)
    cth = np.cos(theta)
    sph = np.sin(phi)
    cph = np.cos(phi)
    sps = np.sin(psi)
    cps = np.cos(psi)

    # Derivatives for vn, ve, vd with respect to vt, alpha, beta
    dvn_dv = ca * cb * cps * cth + sb * (-sps * cph + cps * sth * sph) + sa * cb * (sph * sps + cph * sth * cps)
    dvn_dal = vt * (-sa * cb * cps * cth + ca * cb * (sph * sps + cph * sth * cps))
    dvn_dbt = vt * (-ca * sb * cps * cth + cb * (-sps * cph + cps * sth * sph) - sa * sb * (sph * sps + cph * sth * cps))

    dve_dv = ca * cb * cth * sps + sb * (sph * sth * sps + cph * cps) + sa * cb * (-sph * cps + cph * sth * sps)
    dve_dal = vt * (-sa * cb * cth * sps + ca * cb * (-sph * cps + cph * sth * sps))
    dve_dbt = vt * (-ca * sb * cth * sps + ca * (sph * sth * sps + cph * cps) + sa * sb * (-sph * cps + cph * sth * sps))

    dvd_dv = ca * cb * sth - sb * sph * cth - sa * cb * cph * cth
    dvd_dal = vt * (-sa * cb * sth - ca * cb * cph * cth)
    dvd_dbt = vt * (-ca * sb * sth - cb * sph * cth + sa * sb * cph * cth)

    H = None
    if GPS_setting == 'false': # Changed from 'false_no_h' to 'false'
        if object_setting == 'alpha':
            # H = [d(vt)/d(vt), d(vt)/d(alpha), d(vt)/d(beta), d(vt)/d(h)
            #      d(h)/d(vt), d(h)/d(alpha), d(h)/d(beta), d(h)/d(h)
            #      d(alpha)/d(vt), d(alpha)/d(alpha), d(alpha)/d(beta), d(alpha)/d(h)]
            H = np.array([
                [1, 0, 0],
                [0, 1, 0]
            ])
    elif GPS_setting == 'true': # Changed from 'true_no_h' to 'true'
        if object_setting == 'alpha':
            # H = [d(vt)/d(vt), d(vt)/d(alpha), d(vt)/d(beta), d(vt)/d(h), d(vt)/d(wn), d(vt)/d(we), d(vt)/d(wd)
            #      d(h)/d(vt), d(h)/d(alpha), d(h)/d(beta), d(h)/d(h), d(h)/d(wn), d(h)/d(we), d(h)/d(wd)
            #      d(vn)/d(vt), d(vn)/d(alpha), d(vn)/d(beta), d(vn)/d(h), d(vn)/d(wn), d(vn)/d(we), d(vn)/d(wd)
            #      d(ve)/d(vt), d(ve)/d(alpha), d(ve)/d(beta), d(ve)/d(h), d(ve)/d(wn), d(ve)/d(we), d(ve)/d(wd)
            #      d(vd)/d(vt), d(vd)/d(alpha), d(vd)/d(beta), d(vd)/d(h), d(vd)/d(wn), d(vd)/d(we), d(vd)/d(wd)
            #      d(alpha)/d(vt), d(alpha)/d(alpha), d(alpha)/d(beta), d(alpha)/d(h), d(alpha)/d(wn), d(alpha)/d(we), d(alpha)/d(wd)]
            H = np.array([
                [1, 0, 0, 0, 0, 0],  # vt
                [dvn_dv, dvn_dal, dvn_dbt, 1, 0, 0],  # vn
                [dve_dv, dve_dal, dve_dbt, 0, 1, 0],  # ve
                [dvd_dv, dvd_dal, dvd_dbt, 0, 0, 1],  # vd
                [0, 1, 0, 0, 0, 0]  # alpha
            ])
    return H


