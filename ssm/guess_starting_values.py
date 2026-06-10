# init_guess.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import gruanpy as gp

from ssm.ssm_model import (
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I,
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S,
    R_S, U_S, LU_S, V_S, LV_S
)


def _ensure_obs_shapes(obs: np.ndarray, meas_var: np.ndarray) -> None:
    obs = np.asarray(obs)
    meas_var = np.asarray(meas_var)
    if obs.ndim != 2 or obs.shape[1] != 8:
        raise ValueError("obs must be shape (n, 8)")
    if meas_var.ndim != 2 or meas_var.shape != obs.shape:
        raise ValueError("meas_var must have same shape as obs (n, 8)")


def _propagate_to_thv_variance(T0: float, p0: float, r0: float, meas_var_row: np.ndarray) -> float:
    """
    Propagate measurement variances (T, p, r) to variance of theta_v
    using analytic partial derivatives (first-order error propagation).
    """
    # numeric safeguards
    eps = 1e-12
    p_safe = max(float(p0), eps)
    r_safe = max(float(r0), 0.0)

    A = (gp.p0 / p_safe) ** gp.Poisson_exponent
    B = (1.0 + 0.61 * r_safe)

    # derivatives consistent with virtual potential temperature definition
    dthv_dT = A * B
    dthv_dp = -gp.Poisson_exponent * T0 * A * B / p_safe
    dthv_dr = 0.61 * T0 * A

    var_T = float(meas_var_row[T_I])
    var_p = float(meas_var_row[P_I])
    var_r = float(meas_var_row[R_I])

    thv_var = (dthv_dT ** 2) * var_T + (dthv_dp ** 2) * var_p + (dthv_dr ** 2) * var_r
    # ensure non-negative
    return float(max(thv_var, 1e-12))


def guess_initial_state(obs: np.ndarray, meas_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Guess initial state s0 and covariance P0 from observations.

    Parameters
    ----------
    obs : ndarray (n,8)
        Observations [z, Lz, T, p, RH, r, u, v]
    meas_var : ndarray (n,8)
        Measurement variances

    Returns
    -------
    s0 : ndarray (12,)
        Initial state estimate
    P0 : ndarray (12,12)
        Initial covariance estimate
    """
    obs = np.asarray(obs, dtype=float)
    meas_var = np.asarray(meas_var, dtype=float)

    _ensure_obs_shapes(obs, meas_var)

    n = obs.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations to estimate trends.")

    # first two observations
    o0 = obs[0]
    o1 = obs[1]

    # compute theta_v at first two levels using gruanpy helper
    thv0 = gp.virtual_potential_temperature(o0[T_I], o0[P_I], o0[R_I])
    thv1 = gp.virtual_potential_temperature(o1[T_I], o1[P_I], o1[R_I])
    # trend as simple difference per level (preserve original behavior)
    Lthv0 = float(thv1 - thv0)

    # build initial state vector
    s0 = np.zeros(12, dtype=float)
    s0[Z_S]    = float(o0[Z_I])
    s0[LZ_S]   = float(o0[LZ_I])
    s0[Thv_S]  = float(thv0)
    s0[LThv_S] = float(Lthv0)
    s0[P_S]    = float(o0[P_I])
    s0[RH_S]   = float(o0[RH_I])
    s0[LRH_S]  = float(o1[RH_I] - o0[RH_I])
    s0[R_S]    = float(o0[R_I])
    s0[U_S]    = float(o0[U_I])
    s0[LU_S]   = float(o1[U_I] - o0[U_I])
    s0[V_S]    = float(o0[V_I])
    s0[LV_S]   = float(o1[V_I] - o0[V_I])

    # build initial covariance P0
    P0 = np.eye(12, dtype=float)

    # physical variances from measurement uncertainty (first level)
    P0[Z_S, Z_S]   = float(meas_var[0, Z_I])
    P0[LZ_S, LZ_S] = float(meas_var[0, LZ_I])
    P0[P_S, P_S]   = float(meas_var[0, P_I])
    P0[RH_S, RH_S] = float(meas_var[0, RH_I])
    P0[R_S, R_S]   = float(meas_var[0, R_I])
    P0[U_S, U_S]   = float(meas_var[0, U_I])
    P0[V_S, V_S]   = float(meas_var[0, V_I])

    # propagate variance to theta_v using analytic derivatives
    thv_var = _propagate_to_thv_variance(o0[T_I], o0[P_I], o0[R_I], meas_var[0])
    P0[Thv_S, Thv_S] = thv_var

    # trend variances: moderate default values (tunable)
    P0[LThv_S, LThv_S] = 1.0
    P0[LRH_S,  LRH_S]  = 1.0
    P0[LU_S,   LU_S]   = 1.0
    P0[LV_S,   LV_S]   = 1.0

    # ensure symmetry and numerical safety
    P0 = 0.5 * (P0 + P0.T)
    # small jitter to avoid exact singularities
    P0 += 1e-12 * np.eye(12)

    return s0, P0
