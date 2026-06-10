import numpy as np
from ssm.ssm_model import (
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I,
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S
)
import gruanpy as gp


# ============================================================
# 1. STANDARDIZATION (MIN–MAX)
# ============================================================

def standardize_obs(obs, meas_var, eps=1e-9):
    obs = np.asarray(obs)
    meas_var = np.asarray(meas_var)

    o_min = obs.min(axis=0)
    o_max = obs.max(axis=0)
    rng = o_max - o_min + eps

    obs_std = (obs - o_min) / rng
    meas_var_std = meas_var / (rng**2)

    params = {"min": o_min, "max": o_max, "range": rng}
    return obs_std, meas_var_std, params


# ============================================================
# 2. DENORMALIZATION OF OBSERVATIONS
# ============================================================

def denormalize_obs(obs_std, params):
    o_min = params["min"]
    rng = params["range"]
    return obs_std * rng + o_min


# ============================================================
# 3. RECONSTRUCT PHYSICAL STATES FROM DENORMALIZED OBS
# ============================================================

def reconstruct_physical_states(states, obs_phys):
    """
    Replace the observed components of the state vector with
    denormalized observations and recompute theta_v physically.
    """

    states_phys = states.copy()

    # extract physical obs
    T = np.maximum(obs_phys[:, T_I], 0.1)
    p = np.maximum(obs_phys[:, P_I], 1.0)
    r = np.maximum(obs_phys[:, R_I], 0.0)

    # recompute theta_v
    thv_phys = np.array([
        gp.virtual_potential_temperature(T[i], p[i], r[i])
        for i in range(len(T))
    ])

    # overwrite theta_v
    states_phys[:, Thv_S] = thv_phys

    # overwrite physical components
    states_phys[:, Z_S]  = obs_phys[:, Z_I]
    states_phys[:, LZ_S] = obs_phys[:, LZ_I]
    states_phys[:, P_S]  = obs_phys[:, P_I]
    states_phys[:, RH_S] = obs_phys[:, RH_I]
    states_phys[:, R_S]  = obs_phys[:, R_I]
    states_phys[:, U_S]  = obs_phys[:, U_I]
    states_phys[:, V_S]  = obs_phys[:, V_I]

    # trend components remain unchanged:
    # LThv_S, LRH_S, LU_S, LV_S

    return states_phys
