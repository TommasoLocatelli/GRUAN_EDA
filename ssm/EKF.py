import numpy as np
import statsmodels.api as sm

# ============================================================
# STATE INDICES (14-dimensional state vector)
# s_t = [z_t, Λ_z_t, θv_t, Λ_θv_t, p_t, Λ_p_t,
#        RH_t, Λ_RH_t, r_t, Λ_r_t, u_t, Λ_u_t, v_t, Λ_v_t]^T
# ============================================================
IZ, ILZ     = 0, 1          # z_t, Λ_z_t
IThv, ILThv = 2, 3          # θ_v,t, Λ_θv,t
IP, ILP     = 4, 5          # p_t, Λ_p_t
IRH, ILRH   = 6, 7          # RH_t, Λ_RH_t
Ir, ILr     = 8, 9          # r_t, Λ_r_t
IU, ILU     = 10, 11        # u_t, Λ_u_t
IV, ILV     = 12, 13        # v_t, Λ_v_t

p0 = 1000.0     # reference pressure
kappa = 0.286   # R/cp (adjust if needed)

# ============================================================
# OBSERVATION INDICES
# x_t = [z_t, T_t, p_t, RH_t, r_t, u_t, v_t]^T
# ============================================================
IZ_obs, IT_obs, IP_obs, IRH_obs, Ir_obs, IU_obs, IV_obs = 0, 1, 2, 3, 4, 5, 6


# ============================================================
# NONLINEAR MEASUREMENT FUNCTION A(s_t)
# x_t = [z_t, T_t, p_t, RH_t, r_t, u_t, v_t]^T
# ============================================================
def A(s):
    z = s[IZ]
    thv = s[IThv]
    p = s[IP]
    RH = s[IRH]
    r = s[Ir]
    u = s[IU]
    v = s[IV]

    factor_p = (p / p0) ** kappa
    factor_r = 1.0 / (1.0 + 0.61 * r)
    T = thv * factor_p * factor_r

    return np.array([z, T, p, RH, r, u, v])


# ============================================================
# JACOBIAN J_A(s_t) = ∂A/∂s
# Shape: (7, 14)
# ============================================================
def J_A(s):
    thv = s[IThv]
    p = s[IP]
    r = s[Ir]

    factor_p = (p / p0) ** kappa
    factor_r = 1.0 / (1.0 + 0.61 * r)

    dT_dthv = factor_p * factor_r
    dT_dp   = thv * kappa * (p ** (kappa - 1)) / (p0 ** kappa) * factor_r
    dT_dr   = -0.61 * thv * factor_p * (factor_r ** 2)

    J = np.zeros((7, 14))

    # z_t row
    J[IZ_obs, IZ] = 1.0

    # T_t row
    J[IT_obs, IThv] = dT_dthv
    J[IT_obs, IP]   = dT_dp
    J[IT_obs, Ir]   = dT_dr

    # p_t row
    J[IP_obs, IP] = 1.0

    # RH_t row
    J[IRH_obs, IRH] = 1.0

    # r_t row
    J[Ir_obs, Ir] = 1.0

    # u_t row
    J[IU_obs, IU] = 1.0

    # v_t row
    J[IV_obs, IV] = 1.0

    return J


# ============================================================
# TRANSITION MATRIX Φ FOR LOCAL LINEAR TREND (14x14)
# ============================================================
def build_Phi():
    k_states = 14
    Phi = np.eye(k_states)

    # z, Λ_z
    Phi[IZ, IZ] = 1.0
    Phi[IZ, ILZ] = 1.0
    Phi[ILZ, ILZ] = 1.0

    # θ_v, Λ_θv
    Phi[IThv, IThv] = 1.0
    Phi[IThv, ILThv] = 1.0
    Phi[ILThv, ILThv] = 1.0

    # p, Λ_p
    Phi[IP, IP] = 1.0
    Phi[IP, ILP] = 1.0
    Phi[ILP, ILP] = 1.0

    # RH, Λ_RH
    Phi[IRH, IRH] = 1.0
    Phi[IRH, ILRH] = 1.0
    Phi[ILRH, ILRH] = 1.0

    # r, Λ_r
    Phi[Ir, Ir] = 1.0
    Phi[Ir, ILr] = 1.0
    Phi[ILr, ILr] = 1.0

    # u, Λ_u
    Phi[IU, IU] = 1.0
    Phi[IU, ILU] = 1.0
    Phi[ILU, ILU] = 1.0

    # v, Λ_v
    Phi[IV, IV] = 1.0
    Phi[IV, ILV] = 1.0
    Phi[ILV, ILV] = 1.0

    return Phi


# ============================================================
# EKF LOG-LIKELIHOOD + FILTER
# ============================================================
def ekf_loglike_and_filter(y, Phi, Q, R_seq, s0, P0):
    """
    y     : (T,7) observations [z_t, T_t, p_t, RH_t, r_t, u_t, v_t]
    Phi   : (14,14) transition matrix
    Q     : (14,14) process noise covariance
    R_seq : (7,7,T) measurement noise covariance (time-varying)
    s0    : (14,) initial state
    P0    : (14,14) initial covariance
    """
    T = y.shape[0]
    k_states = s0.shape[0]
    k_endog = y.shape[1]

    s_filt = np.zeros((T, k_states))
    P_filt = np.zeros((T, k_states, k_states))

    s_pred = s0.copy()
    P_pred = P0.copy()

    loglike = 0.0
    const = k_endog * np.log(2.0 * np.pi)

    for t in range(T):
        # Prediction
        s_pred = Phi @ s_pred
        P_pred = Phi @ P_pred @ Phi.T + Q

        # Linearization
        H_t = J_A(s_pred)
        y_pred = A(s_pred)
        R_t = R_seq[:, :, t]

        S_t = H_t @ P_pred @ H_t.T + R_t
        sign, logdet = np.linalg.slogdet(S_t)
        if sign <= 0:
            return -1e10, s_filt, P_filt  # penalize non-PD

        K_t = P_pred @ H_t.T @ np.linalg.inv(S_t)
        innov = y[t] - y_pred

        # Log-likelihood contribution
        ll_t = -0.5 * (const + logdet + innov.T @ np.linalg.inv(S_t) @ innov)
        loglike += ll_t

        # Update
        s_upd = s_pred + K_t @ innov
        P_upd = (np.eye(k_states) - K_t @ H_t) @ P_pred

        s_filt[t] = s_upd
        P_filt[t] = P_upd

        s_pred, P_pred = s_upd, P_upd

    return loglike, s_filt, P_filt


# ============================================================
# MLEModel SUBCLASS WITH EKF
# ============================================================
class NonlinearThermoTrendEKF(sm.tsa.statespace.MLEModel):
    """
    Nonlinear local-trend SSM with thermodynamic T_t in the measurement.
    endog: (T,7) array [z_t, T_t, p_t, RH_t, r_t, u_t, v_t]
    meas_var_*: (T,) arrays of measurement variances for each observed component.
    """

    def __init__(self,
                 endog,
                 meas_var_z,
                 meas_var_T,
                 meas_var_p,
                 meas_var_RH,
                 meas_var_r,
                 meas_var_u,
                 meas_var_v):
        k_states = 14
        k_posdef = 14

        endog = np.asarray(endog)
        self.meas_var_z  = np.asarray(meas_var_z)
        self.meas_var_T  = np.asarray(meas_var_T)
        self.meas_var_p  = np.asarray(meas_var_p)
        self.meas_var_RH = np.asarray(meas_var_RH)
        self.meas_var_r  = np.asarray(meas_var_r)
        self.meas_var_u  = np.asarray(meas_var_u)
        self.meas_var_v  = np.asarray(meas_var_v)

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="known",
        )

        # Transition and selection
        self.Phi = build_Phi()
        self.ssm["transition"] = self.Phi
        self.ssm["selection"] = np.eye(k_states)

        # Dummy linear design (we don't use the built-in linear filter)
        self.ssm["design"] = np.zeros((7, k_states))

        # Time-varying obs_cov (R_seq) for EKF
        self.R_seq = np.zeros((7, 7, self.nobs))
        self.R_seq[IZ_obs, IZ_obs, :]   = self.meas_var_z
        self.R_seq[IT_obs, IT_obs, :]   = self.meas_var_T
        self.R_seq[IP_obs, IP_obs, :]   = self.meas_var_p
        self.R_seq[IRH_obs, IRH_obs, :] = self.meas_var_RH
        self.R_seq[Ir_obs, Ir_obs, :]   = self.meas_var_r
        self.R_seq[IU_obs, IU_obs, :]   = self.meas_var_u
        self.R_seq[IV_obs, IV_obs, :]   = self.meas_var_v

        # Initial state and covariance for EKF
        self.initial_state = np.zeros(k_states)
        self.initial_state_cov = np.eye(k_states) * 1e4

        # Cache index for state_cov diagonal
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------
    @property
    def param_names(self):
        return [
            "sigma2_z", "sigma2_Lz",
            "sigma2_thv", "sigma2_Lthv",
            "sigma2_p", "sigma2_Lp",
            "sigma2_RH", "sigma2_LRH",
            "sigma2_r", "sigma2_Lr",
            "sigma2_u", "sigma2_Lu",
            "sigma2_v", "sigma2_Lv",
        ]

    @property
    def start_params(self):
        return np.ones(14) * np.var(self.endog, axis=0).mean()

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return np.sqrt(constrained)

    # --------------------------------------------------------
    # UPDATE: build Q from params (diagonal state_cov)
    # --------------------------------------------------------
    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)
        self.ssm[self._state_cov_idx] = params
        return params

    # --------------------------------------------------------
    # LOGLIKE USING EKF
    # --------------------------------------------------------
    def loglike(self, params, *args, **kwargs):
        params = self.transform_params(np.array(params))
        Q = np.diag(params)

        s0 = self.initial_state
        P0 = self.initial_state_cov

        ll, _, _ = ekf_loglike_and_filter(
            self.endog,
            self.Phi,
            Q,
            self.R_seq,
            s0,
            P0,
        )
        return ll

    # --------------------------------------------------------
    # FILTERED STATES VIA EKF
    # --------------------------------------------------------
    def ekf_filter(self, params):
        params = self.transform_params(np.array(params))
        Q = np.diag(params)
        s0 = self.initial_state
        P0 = self.initial_state_cov

        _, s_filt, P_filt = ekf_loglike_and_filter(
            self.endog,
            self.Phi,
            Q,
            self.R_seq,
            s0,
            P0,
        )
        return s_filt, P_filt


# ============================================================
# EXAMPLE USAGE
# ============================================================
# y: (T,7) array with columns [z_t, T_t, p_t, RH_t, r_t, u_t, v_t]
# meas_var_*: (T,) arrays of measurement variances
#
# model = NonlinearThermoTrendEKF(
#     y,
#     meas_var_z,
#     meas_var_T,
#     meas_var_p,
#     meas_var_RH,
#     meas_var_r,
#     meas_var_u,
#     meas_var_v,
# )
# res = model.fit(method="bfgs")
# s_filt, P_filt = model.ekf_filter(res.params)
