# ssm_scaling.py
from __future__ import annotations
from typing import Tuple, Dict
from pathlib import Path
import numpy as np
import gruanpy as gp

from ssm.ssm_model import (
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I,
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S
)


def _check_obs_shapes(obs: np.ndarray, meas_var: np.ndarray) -> None:
    obs = np.asarray(obs)
    meas_var = np.asarray(meas_var)
    if obs.ndim != 2 or obs.shape[1] != 8:
        raise ValueError("obs must be shape (n, 8)")
    if meas_var.ndim != 2 or meas_var.shape != obs.shape:
        raise ValueError("meas_var must have same shape as obs (n, 8)")


def standardize_obs(obs: np.ndarray, meas_var: np.ndarray, eps: float = 1e-9
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Min–max standardization of observations and measurement variances.

    Parameters
    ----------
    obs : ndarray (n,8)
        Observations in order [z, Lz, T, p, RH, r, u, v]
    meas_var : ndarray (n,8)
        Measurement variances for each observation component
    eps : float
        Small value added to denominators to avoid division by zero

    Returns
    -------
    obs_std : ndarray (n,8)
        Standardized observations in [0,1]
    meas_var_std : ndarray (n,8)
        Standardized measurement variances (scaled by range^2)
    params : dict
        Dictionary with keys "min", "max", "range" each an ndarray (8,)
    """
    obs = np.asarray(obs, dtype=float)
    meas_var = np.asarray(meas_var, dtype=float)

    _check_obs_shapes(obs, meas_var)

    o_min = np.nanmin(obs, axis=0)
    o_max = np.nanmax(obs, axis=0)
    rng = (o_max - o_min)
    rng = np.where(rng <= 0.0, eps, rng)  # avoid zero range

    obs_std = (obs - o_min) / rng
    meas_var_std = meas_var / (rng ** 2)

    params = {"min": o_min, "max": o_max, "range": rng}
    return obs_std, meas_var_std, params


def denormalize_obs(obs_std: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Undo min–max standardization.

    Parameters
    ----------
    obs_std : ndarray (n,8)
    params : dict with keys "min" and "range"

    Returns
    -------
    obs_phys : ndarray (n,8)
    """
    obs_std = np.asarray(obs_std, dtype=float)
    o_min = np.asarray(params["min"], dtype=float)
    rng = np.asarray(params["range"], dtype=float)

    if obs_std.ndim == 1:
        obs_std = obs_std.reshape(1, -1)

    if obs_std.shape[1] != 8 or o_min.shape[0] != 8 or rng.shape[0] != 8:
        raise ValueError("Shapes mismatch: obs_std must be (n,8) and params must contain 8-element arrays")

    return obs_std * rng + o_min


def denormalize_states(states_std: np.ndarray, obs_phys: np.ndarray) -> np.ndarray:
    """
    Denormalize state vectors and recompute theta_v from physical observations.

    Parameters
    ----------
    states_std : ndarray (n,12)
        Standardized states (same ordering as your model)
    obs_phys : ndarray (n,8)
        Denormalized observations (physical units)

    Returns
    -------
    states_phys : ndarray (n,12)
        States with physical components overwritten from obs_phys and Thv recomputed
    """
    states_std = np.asarray(states_std, dtype=float)
    obs_phys = np.asarray(obs_phys, dtype=float)

    if states_std.ndim != 2 or states_std.shape[1] != 12:
        raise ValueError("states_std must be shape (n, 12)")
    if obs_phys.ndim != 2 or obs_phys.shape[1] != 8:
        raise ValueError("obs_phys must be shape (n, 8)")
    if states_std.shape[0] != obs_phys.shape[0]:
        raise ValueError("states_std and obs_phys must have the same number of rows")

    states_phys = states_std.copy()

    # vectorized recomputation of theta_v
    T = obs_phys[:, T_I]
    p = obs_phys[:, P_I]
    r = obs_phys[:, R_I]

    # gp.virtual_potential_temperature may accept scalars only; vectorize safely
    vec_thv = np.vectorize(lambda T_, p_, r_: gp.virtual_potential_temperature(T_, p_, r_))
    thv_phys = vec_thv(T, p, r).astype(float)

    # overwrite theta_v and physical components
    states_phys[:, Thv_S] = thv_phys
    states_phys[:, Z_S]  = obs_phys[:, Z_I]
    states_phys[:, LZ_S] = obs_phys[:, LZ_I]
    states_phys[:, P_S]  = obs_phys[:, P_I]
    states_phys[:, RH_S] = obs_phys[:, RH_I]
    states_phys[:, R_S]  = obs_phys[:, R_I]
    states_phys[:, U_S]  = obs_phys[:, U_I]
    states_phys[:, V_S]  = obs_phys[:, V_I]

    # trend components (LThv_S, LRH_S, LU_S, LV_S) are left unchanged
    return states_phys
