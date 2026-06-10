import numpy as np
from numpy import eye, array
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp

# ============================================================
# STATE VECTOR STRUCTURE
# s_t = [z, Lz, Thv, LThv, p, RH, LRH, r, u, Lu, v, Lv]^T
# ============================================================

Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S = range(12)

# ============================================================
# STATE TRANSITION MATRIX (LOCAL LEVEL + LOCAL TREND)
# ============================================================

PHI = eye(12)

# local trend couplings
PHI[Z_S,    LZ_S]    = 1.0
PHI[Thv_S,  LThv_S]  = 1.0
PHI[RH_S,   LRH_S]   = 1.0
PHI[U_S,    LU_S]    = 1.0
PHI[V_S,    LV_S]    = 1.0

# ============================================================
# OBSERVATION VECTOR STRUCTURE
# x_t = [z, Lz, T, p, RH, r, u, v]^T
# ============================================================

Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I = range(8)

# ============================================================
# NONLINEAR MEASUREMENT FUNCTION A(s_t)
# Maps state → observation
# ============================================================

def A(s):
    # extract scalars (avoid numpy scalar warnings)
    z   = float(np.squeeze(s[Z_S]))
    Lz  = float(np.squeeze(s[LZ_S]))
    thv = float(np.squeeze(s[Thv_S]))
    p   = float(np.squeeze(s[P_S]))
    RH  = float(np.squeeze(s[RH_S]))
    r   = float(np.squeeze(s[R_S]))
    u   = float(np.squeeze(s[U_S]))
    v   = float(np.squeeze(s[V_S]))

    # invert virtual potential temperature
    T = gp.virtual_potential_temperature_inverse(thv, p, r)

    return array([z, Lz, T, p, RH, r, u, v])


# ============================================================
# JACOBIAN OF A(s_t)
# J_A(s) = ∂A/∂s
# Shape: (8 × 12)
# ============================================================

def J_A(s):
    p0    = gp.p0
    kappa = gp.Poisson_exponent

    # extract scalars
    thv = float(np.squeeze(s[Thv_S]))
    p   = float(np.squeeze(s[P_S]))
    r   = float(np.squeeze(s[R_S]))

    # partial derivatives of T(thv, p, r)
    factor_p = (p / p0) ** kappa
    factor_r = 1.0 / (1.0 + 0.61 * r)

    dT_dthv = factor_p * factor_r
    dT_dp   = thv * kappa * (p ** (kappa - 1)) / (p0 ** kappa) * factor_r
    dT_dr   = -0.61 * thv * factor_p * (factor_r ** 2)

    # initialize Jacobian
    J = np.zeros((8, 12))

    # z
    J[Z_I, Z_S] = 1.0

    # T
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

    return J
