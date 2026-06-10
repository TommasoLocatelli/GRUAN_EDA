# ssm_model.py
from __future__ import annotations
import numpy as np
from numpy import eye, array
from typing import Sequence

# ============================================================
# STATE VECTOR STRUCTURE
# s_t = [z, Lz, Thv, LThv, p, RH, LRH, r, u, Lu, v, Lv]^T
# ============================================================
Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S = range(12)

# ============================================================
# OBSERVATION VECTOR STRUCTURE
# x_t = [z, Lz, T, p, RH, r, u, v]^T
# ============================================================
Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I = range(8)

# ============================================================
# STATE TRANSITION MATRIX (LOCAL LEVEL + LOCAL TREND)
# ============================================================
PHI = eye(12)
PHI[Z_S,    LZ_S]    = 1.0
PHI[Thv_S,  LThv_S]  = 1.0
PHI[RH_S,   LRH_S]   = 1.0
PHI[U_S,    LU_S]    = 1.0
PHI[V_S,    LV_S]    = 1.0

# ============================================================
# External dependency: gruanpy (must provide p0 and Poisson_exponent)
# ============================================================
try:
    import gruanpy as gp
except Exception as exc:
    raise ImportError("gruanpy is required by ssm_model.py — install it or mock gp for tests") from exc

# Validate gp constants
if not hasattr(gp, "p0") or not hasattr(gp, "Poisson_exponent"):
    raise ImportError("gruanpy must expose 'p0' and 'Poisson_exponent' attributes")

# ============================================================
# NONLINEAR MEASUREMENT FUNCTION A(s_t)
# Maps state → observation
# ============================================================
def A(s: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Measurement function: state vector -> observation vector (length 8).
    s: length-12 state vector
    returns: ndarray shape (8,)
    """
    s = np.asarray(s, dtype=float).reshape(-1)
    assert s.size == 12, "state vector must have length 12"

    # extract scalars
    z   = float(s[Z_S])
    Lz  = float(s[LZ_S])
    thv = float(s[Thv_S])
    p   = float(s[P_S])
    RH  = float(s[RH_S])
    r   = float(s[R_S])
    u   = float(s[U_S])
    v   = float(s[V_S])

    # invert virtual potential temperature (delegate to gp)
    T = gp.virtual_potential_temperature_inverse(thv, p, r)

    out = np.array([z, Lz, T, p, RH, r, u, v], dtype=float)
    assert out.shape == (8,)
    return out

# ============================================================
# JACOBIAN OF A(s_t)
# J_A(s) = ∂A/∂s
# Shape: (8 × 12)
# ============================================================
def J_A(s: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Jacobian matrix of A with respect to state s.
    returns ndarray shape (8,12)
    """
    s = np.asarray(s, dtype=float).reshape(-1)
    assert s.size == 12, "state vector must have length 12"

    p0    = float(gp.p0)
    kappa = float(gp.Poisson_exponent)

    thv = float(s[Thv_S])
    p   = float(s[P_S])
    r   = float(s[R_S])

    # numeric safeguards
    eps = 1e-12
    p_safe = max(p, eps)
    r_safe = max(r, 0.0)  # mixing ratio should be >= 0

    factor_p = (p_safe / p0) ** kappa
    factor_r = 1.0 / (1.0 + 0.61 * r_safe)

    dT_dthv = factor_p * factor_r
    # dT_dp: careful with powers; compute using safe values
    dT_dp = thv * kappa * (p_safe ** (kappa - 1)) / (p0 ** kappa) * factor_r
    dT_dr = -0.61 * thv * factor_p * (factor_r ** 2)

    J = np.zeros((8, 12), dtype=float)

    # z
    J[Z_I, Z_S] = 1.0

    # Lz
    J[LZ_I, LZ_S] = 1.0

    # T partials
    J[T_I, Thv_S] = dT_dthv
    J[T_I, P_S]   = dT_dp
    J[T_I, R_S]   = dT_dr

    # p
    J[P_I, P_S] = 1.0

    # RH
    J[RH_I, RH_S] = 1.0

    # r
    J[R_I, R_S] = 1.0

    # u
    J[U_I, U_S] = 1.0

    # v
    J[V_I, V_S] = 1.0

    assert J.shape == (8, 12)
    return J
