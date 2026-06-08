import numpy as np
from numpy import eye, array, asarray
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import gruanpy as gp

def prep_ekf(path, upper_bound=3000, Q_scale=1000):

    gdp=gp.read(path)
    upper_bound=gp._find_upper_bound(gdp.data[['alt']], upper_bound=upper_bound, return_value=True) # find the PBLH upper bound for profile
    data = gdp.data[gdp.data['alt'] <= upper_bound]  # Limit to first 3.5 km

    def observations(data):
        z = data['alt'].values
        z_var = (data['alt_gph_uc'].values * 0.5)**2
        Lz = data['vspeed'].values
        Lz_var = (data['vspeed_uc'].values * 0.5)**2
        T = data['temp'].values
        T_var = (data['temp_uc'].values * 0.5)**2
        p = data['press'].values
        p_var = (data['press_uc'].values * 0.5)**2
        rh = data['rh'].values
        rh_var = (data['rh_uc'].values * 0.5)**2
        r = data['wvmr_mass'].values
        r_var = (data['wvmr_mass_uc'].values * 0.5)**2
        u = data['wzon'].values
        u_var = (data['wzon_uc'].values * 0.5)**2
        v = data['wmeri'].values
        v_var = (data['wmeri_uc'].values * 0.5)**2
        obs =[np.array([z[i], Lz[i], T[i], p[i], rh[i], r[i], u[i], v[i]]).reshape(-1, 1) for i in range(len(z))]
        meas_var = [np.array([z_var[i], Lz_var[i], T_var[i], p_var[i], rh_var[i], r_var[i], u_var[i], v_var[i]]).reshape(-1, 1) for i in range(len(z))]

        return obs, meas_var

    obs, meas_var = observations(data)

    # obs vector
    # x_t = [z_t, Lz_t, T_t, p_t, RH_t, r_t, u_t, v_t]
    # obs indexes
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I = range(8)

    # state vector
    # s_t = [z_t, Lz_t, Thv_t, LThv_t, p_t, RH_t, LRH_t, r_t, u_t, Lu_t, v_t, Lv_t]
    # state indexes
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S = range(12)

    # state transition matrix
    Phi = np.eye(12) # local level state transition matrix
    Phi[Z_S, LZ_S] = 1 # local trend components
    Phi[Thv_S, LThv_S] = 1
    Phi[RH_S, LRH_S] = 1
    Phi[U_S, LU_S] = 1
    Phi[V_S, LV_S] = 1

    # innovation w_t are iid zero mean normal vectors with covariance matrix Q
    Q = eye(12) * Q_scale # process noise covariance matrix

    # ============================================================
    # NONLINEAR MEASUREMENT FUNCTION A(s_t)
    # x_t = [z_t, Lz_t, T_t, p_t, RH_t, r_t, u_t, v_t]^T
    # ============================================================

    p0 = 1000.0     # reference pressure
    kappa = gp.Poisson_exponent

    def A(s):
        # ensure we work with Python scalars not 1-element arrays to avoid
        # numpy deprecation warnings when constructing outputs
        z = float(np.squeeze(s[Z_S]))
        Lz = float(np.squeeze(s[LZ_S]))
        thv = float(np.squeeze(s[Thv_S]))
        p = float(np.squeeze(s[P_S]))
        RH = float(np.squeeze(s[RH_S]))
        r = float(np.squeeze(s[R_S]))
        u = float(np.squeeze(s[U_S]))
        v = float(np.squeeze(s[V_S]))

        T = gp.virtual_potential_temperature_inverse(thv, p, r)

        return np.array([z, Lz, T, p, RH, r, u, v])


    # ============================================================
    # JACOBIAN J_A(s_t) = ∂A/∂s
    # Shape: (8, 12)
    # ============================================================
    def J_A(s):
        # extract scalars to avoid array-to-scalar deprecation and invalid power ops
        thv = float(np.squeeze(s[Thv_S]))
        p = float(np.squeeze(s[P_S]))
        r = float(np.squeeze(s[R_S]))

        factor_p = (p / p0) ** kappa
        factor_r = 1.0 / (1.0 + 0.61 * r)
        dT_dthv = factor_p * factor_r
        dT_dp   = thv * kappa * (p ** (kappa - 1)) / (p0 ** kappa) * factor_r
        dT_dr   = -0.61 * thv * factor_p * (factor_r ** 2)

        J = np.zeros((8, 12))

        # z_t row
        J[Z_I, Z_S] = 1.0

        # T_t row
        J[T_I, Thv_S] = dT_dthv
        J[T_I, P_S]   = dT_dp
        J[T_I, R_S]   = dT_dr

        # p_t row
        J[P_I, P_S] = 1.0

        # RH_t row
        J[RH_I, RH_S] = 1.0

        # r_t row
        J[R_I, R_S] = 1.0

        # u_t row
        J[U_I, U_S] = 1.0

        # v_t row
        J[V_I, V_S] = 1.0

        return J


    # initial state estimate
    s_0 = array([obs[0][0], # z
            obs[0][1], # Λ_z
            gp.virtual_potential_temperature(obs[0][2], obs[0][3], obs[0][5]), # θ_v
            gp.virtual_potential_temperature(obs[1][2], obs[1][3], obs[1][5]) - gp.virtual_potential_temperature(obs[0][2], obs[0][3], obs[0][5]), # Λ_θv
            obs[0][3], # p
            obs[0][4], # RH
            obs[1][4] - obs[0][4], # Λ_RH
            obs[0][5], # r
            obs[0][6], # u
            obs[1][6] - obs[0][6], # Λ_u
            obs[0][7], # v
            obs[1][7] - obs[0][7]]).reshape(-1, 1)

    # initial error covariance matrix
    P_0 = eye(12) * 1 

    return obs, meas_var, Phi, Q, A, J_A, s_0, P_0
