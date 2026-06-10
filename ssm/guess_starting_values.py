import numpy as np
import gruanpy as gp

from ssm.ssm_model import (
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I,
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S,
    R_S, U_S, LU_S, V_S, LV_S
)


def guess_initial_state(obs, meas_var):
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

    n = obs.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations to estimate trends.")

    # ---------------------------------------------------------
    # Extract first two observations
    # ---------------------------------------------------------
    o0 = obs[0]
    o1 = obs[1]

    # ---------------------------------------------------------
    # Compute theta_v and its trend
    # ---------------------------------------------------------
    thv0 = gp.virtual_potential_temperature(o0[T_I], o0[P_I], o0[R_I])
    thv1 = gp.virtual_potential_temperature(o1[T_I], o1[P_I], o1[R_I])
    Lthv0 = thv1 - thv0

    # ---------------------------------------------------------
    # Build initial state vector
    # ---------------------------------------------------------
    s0 = np.zeros(12)

    s0[Z_S]    = o0[Z_I]
    s0[LZ_S]   = o0[LZ_I]
    s0[Thv_S]  = thv0
    s0[LThv_S] = Lthv0
    s0[P_S]    = o0[P_I]
    s0[RH_S]   = o0[RH_I]
    s0[LRH_S]  = o1[RH_I] - o0[RH_I]
    s0[R_S]    = o0[R_I]
    s0[U_S]    = o0[U_I]
    s0[LU_S]   = o1[U_I] - o0[U_I]
    s0[V_S]    = o0[V_I]
    s0[LV_S]   = o1[V_I] - o0[V_I]

    # ---------------------------------------------------------
    # Build initial covariance P0
    # ---------------------------------------------------------
    P0 = np.eye(12)

    # physical variances from measurement uncertainty
    P0[Z_S, Z_S]   = meas_var[0, Z_I]
    P0[LZ_S, LZ_S] = meas_var[0, LZ_I]
    P0[P_S, P_S]   = meas_var[0, P_I]
    P0[RH_S, RH_S] = meas_var[0, RH_I]
    P0[R_S, R_S]   = meas_var[0, R_I]
    P0[U_S, U_S]   = meas_var[0, U_I]
    P0[V_S, V_S]   = meas_var[0, V_I]

    # ---------------------------------------------------------
    # Propagate variance to theta_v using forward derivatives
    # ---------------------------------------------------------
    T0 = o0[T_I]
    p0 = o0[P_I]
    r0 = o0[R_I]

    A = (gp.p0 / p0) ** gp.Poisson_exponent
    B = (1.0 + 0.61 * r0)

    dthv_dT = A * B
    dthv_dp = -gp.Poisson_exponent * T0 * A * B / p0
    dthv_dr = 0.61 * T0 * A

    thv_var = (
        dthv_dT**2 * meas_var[0, T_I] +
        dthv_dp**2 * meas_var[0, P_I] +
        dthv_dr**2 * meas_var[0, R_I]
    )

    P0[Thv_S, Thv_S] = thv_var

    # trend variances: set to moderate values
    P0[LThv_S, LThv_S] = 1.0
    P0[LRH_S,  LRH_S]  = 1.0
    P0[LU_S,   LU_S]   = 1.0
    P0[LV_S,   LV_S]   = 1.0

    return s0, P0
