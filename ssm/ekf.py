# ekf.py
import numpy as np
from typing import Optional, Tuple
from model import Model

class EKF:
    """
    EKF class: prediction, update, filter, RTS smoother, simulation, likelihood.
    Designed to be instantiated with a Model and observation data.
    """

    def __init__(self, model: Model, obs: np.ndarray, meas_var: np.ndarray, use_pinv: Optional[bool] = None):
        self.model = model
        self.obs = np.asarray(obs)
        self.n = self.obs.shape[0]
        self.q = self.obs.shape[1]
        self.p = self.model.s_0.shape[0]

        # measurement noise covariance R_t (n,q,q)
        self.R = np.zeros((self.n, self.q, self.q))
        for t in range(self.n):
            self.R[t] = np.diag(meas_var[t])

        if use_pinv is not None:
            self.model.use_pinv = use_pinv

        # filter state (runtime)
        self.reset()

    # -------------------------
    # reset / reinit
    # -------------------------
    def reset(self):
        self.s_pred = None
        self.p_pred = None
        self.s_upd = None
        self.p_upd = None
        self.K = None

        # history
        self.s_pred_hist = np.zeros((self.n, self.p))
        self.p_pred_hist = np.zeros((self.n, self.p, self.p))
        self.s_upd_hist = np.zeros((self.n, self.p))
        self.p_upd_hist = np.zeros((self.n, self.p, self.p))
        self.gains_hist = np.zeros((self.n, self.p, self.q))

        # smoother outputs
        self.smooth_s_hist = None
        self.smooth_p_hist = None
        self.smooth_gains_hist = None
        self.lag_one_cov_hist = None

        # convenience
        self.log_likelihood = None

    # -------------------------
    # dimension checks
    # -------------------------
    def _check_dimensions(self):
        assert self.model.Phi.shape == (self.p, self.p)
        assert self.model.Q.shape == (self.p, self.p)
        assert self.model.P_0.shape == (self.p, self.p)
        assert self.obs.shape == (self.n, self.q)
        assert self.R.shape == (self.n, self.q, self.q)

    # -------------------------
    # predict / update / step
    # -------------------------
    def predict(self):
        I = np.eye(self.p)
        Phi = self.model.Phi
        Q = self.model.Q

        if self.s_upd is None:
            s_prev = self.model.s_0
            P_prev = self.model.P_0
        else:
            s_prev = self.s_upd
            P_prev = self.p_upd

        self.s_pred = Phi @ s_prev
        self.s_pred = self.model.apply_state_constraints(self.s_pred)
        self.p_pred = Phi @ P_prev @ Phi.T + Q

        self.p_pred = self.model.ensure_pd(self.p_pred)

    def update(self, x: np.ndarray, R: np.ndarray):
        I = np.eye(self.p)
        J = self.model.J(self.s_pred)           # (q,p)
        x_pred = self.model.A(self.s_pred).reshape(self.q)

        S = J @ self.p_pred @ J.T + R
        S = 0.5 * (S + S.T) + self.model.jitter * np.eye(self.q)

        if self.model.use_pinv:
            S_inv = np.linalg.pinv(S)
        else:
            S_inv = np.linalg.inv(S)

        self.K = self.p_pred @ J.T @ S_inv

        innovation = x - x_pred
        self.s_upd = self.s_pred + self.K @ innovation
        self.s_upd = self.model.apply_state_constraints(self.s_upd)

        IKH = I - self.K @ J
        self.p_upd = IKH @ self.p_pred @ IKH.T + self.K @ R @ self.K.T
        self.p_upd = self.model.ensure_pd(self.p_upd)

    def step(self, x: np.ndarray, R: np.ndarray):
        self.predict()
        self.update(x, R)

    # -------------------------
    # full filter
    # -------------------------
    def filter(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.reset()
        self._check_dimensions()

        for t in range(self.n):
            self.step(self.obs[t], self.R[t])
            self.s_pred_hist[t] = self.s_pred
            self.p_pred_hist[t] = self.p_pred
            self.s_upd_hist[t] = self.s_upd
            self.p_upd_hist[t] = self.p_upd
            self.gains_hist[t] = self.K

        return (self.s_pred_hist, self.p_pred_hist, self.s_upd_hist, self.p_upd_hist, self.gains_hist)

    # -------------------------
    # RTS smoother
    # -------------------------
    def smooth(self):
        if self.s_pred_hist is None or np.all(self.s_pred_hist == 0):
            self.filter()

        n, p = self.n, self.p
        I = np.eye(p)

        s_smooth = np.zeros((n, p))
        P_smooth = np.zeros((n, p, p))
        J_smooth = np.zeros((n - 1, p, p))
        lag_one_cov = np.zeros((n - 1, p, p))

        s_smooth[-1] = self.s_upd_hist[-1]
        P_smooth[-1] = self.p_upd_hist[-1]

        for t in range(n - 2, -1, -1):
            P_upd = self.p_upd_hist[t]
            P_pred_next = self.p_pred_hist[t + 1]
            P_pred_next = self.model.ensure_pd(P_pred_next)

            if self.model.use_pinv:
                inv_Ppred = np.linalg.pinv(P_pred_next)
            else:
                inv_Ppred = np.linalg.inv(P_pred_next)

            J = P_upd @ self.model.Phi.T @ inv_Ppred
            J_smooth[t] = J

            s_smooth[t] = self.s_upd_hist[t] + J @ (s_smooth[t + 1] - self.s_pred_hist[t + 1])
            P_smooth[t] = P_upd + J @ (P_smooth[t + 1] - P_pred_next) @ J.T
            P_smooth[t] = self.model.ensure_pd(P_smooth[t])

            lag_one_cov[t] = J @ P_smooth[t + 1]

        for t in range(n):
            s_smooth[t] = self.model.apply_state_constraints(s_smooth[t])

        self.smooth_s_hist = s_smooth
        self.smooth_p_hist = P_smooth
        self.smooth_gains_hist = J_smooth
        self.lag_one_cov_hist = lag_one_cov

        return s_smooth, P_smooth, J_smooth, lag_one_cov

    # -------------------------
    # simulation smoother
    # -------------------------
    def simulate_states(self):
        if self.smooth_s_hist is None:
            raise RuntimeError("Run smooth() first.")

        n, p = self.n, self.p
        states = np.zeros((n, p))

        states[-1] = np.random.multivariate_normal(self.smooth_s_hist[-1], self.smooth_p_hist[-1])

        for t in range(n - 2, -1, -1):
            P_t = self.smooth_p_hist[t]
            P_tp1 = self.smooth_p_hist[t + 1]
            P_pred_next = self.p_pred_hist[t + 1]
            J = P_t @ self.model.Phi.T @ np.linalg.inv(P_pred_next)

            mean = self.smooth_s_hist[t] + J @ (states[t + 1] - self.smooth_s_hist[t + 1])
            cov = P_t - J @ P_tp1 @ J.T
            cov = self.model.ensure_pd(cov)

            states[t] = np.random.multivariate_normal(mean, cov)

        return states

    # -------------------------
    # simulate observations
    # -------------------------
    def simulate_observations(self, states: np.ndarray) -> np.ndarray:
        n, q = self.n, self.q
        obs_sim = np.zeros((n, q))

        for t in range(n):
            mean = self.model.A(states[t])
            noise = np.random.multivariate_normal(np.zeros(q), self.R[t])
            obs_sim[t] = mean + noise

        return obs_sim

    # -------------------------
    # data likelihood
    # -------------------------
    def data_likelihood(self) -> float:
        if self.s_pred_hist is None or np.all(self.s_pred_hist == 0):
            self.filter()

        ll = 0.0
        for t in range(self.n):
            x_pred = self.s_pred_hist[t]          # (p,)
            P_pred = self.p_pred_hist[t]          # (p,p)
            R = self.R[t]                         # (q,q)

            h_pred = self.model.A(x_pred).reshape(self.q)   # (q,)
            J = self.model.J(x_pred)                        # (q,p)

            Sigma = J @ P_pred @ J.T + R              # (q,q)
            residual = self.obs[t] - h_pred           # (q,)

            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                raise np.linalg.LinAlgError(f"Non-PD innovation covariance at t={t}")

            quad = residual.T @ np.linalg.solve(Sigma, residual)
            ll += -0.5 * (logdet + quad + self.q * np.log(2*np.pi))

        self.log_likelihood = float(ll)
        return self.log_likelihood
