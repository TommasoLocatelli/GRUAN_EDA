import numpy as np

Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I = range(8)

def normalize_ekf_inputs(obs, meas_var, method="minmax", eps=1e-9):
    obs = np.asarray(obs)
    meas_var = np.asarray(meas_var)

    obs_norm = obs.copy().astype(float)
    meas_var_norm = meas_var.copy().astype(float)
    params = {}

    # ---- 1. ALTITUDE + VERTICAL VELOCITY (COUPLED) ----
    z = obs[:, Z_I]
    Lz = obs[:, LZ_I]

    if method == "minmax":
        z_min = np.min(z)
        z_max = np.max(z)
        rng = z_max - z_min + eps

        # altitude
        obs_norm[:, Z_I] = (z - z_min) / rng
        meas_var_norm[:, Z_I] = meas_var[:, Z_I] / (rng**2)

        # velocity: same scale, no shift
        obs_norm[:, LZ_I] = Lz / rng
        meas_var_norm[:, LZ_I] = meas_var[:, LZ_I] / (rng**2)

        params["z"] = {"min": z_min, "max": z_max, "range": rng}

    elif method == "zscore":
        z_mu = np.mean(z)
        z_sigma = np.std(z) + eps

        obs_norm[:, Z_I] = (z - z_mu) / z_sigma
        meas_var_norm[:, Z_I] = meas_var[:, Z_I] / (z_sigma**2)

        obs_norm[:, LZ_I] = Lz / z_sigma
        meas_var_norm[:, LZ_I] = meas_var[:, LZ_I] / (z_sigma**2)

        params["z"] = {"mu": z_mu, "sigma": z_sigma}

    else:
        raise ValueError("Unsupported method for now")

    # ---- 2. OTHER VARIABLES (T, p, RH, r, u, v) ----
    other_idx = [T_I, P_I, RH_I, R_I, U_I, V_I]
    other = obs[:, other_idx]
    other_var = meas_var[:, other_idx]

    if method == "minmax":
        o_min = np.min(other, axis=0)
        o_max = np.max(other, axis=0)
        rng = o_max - o_min + eps

        obs_norm[:, other_idx] = (other - o_min) / rng
        meas_var_norm[:, other_idx] = other_var / (rng**2)

        params["other"] = {"min": o_min, "max": o_max, "range": rng}

    elif method == "zscore":
        mu = np.mean(other, axis=0)
        sigma = np.std(other, axis=0) + eps

        obs_norm[:, other_idx] = (other - mu) / sigma
        meas_var_norm[:, other_idx] = other_var / (sigma**2)

        params["other"] = {"mu": mu, "sigma": sigma}

    return obs_norm, meas_var_norm, params
