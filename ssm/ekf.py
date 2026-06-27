import numpy as np
from typing import Callable, Optional
from ssm.standardize import standardize_obs, denormalize_obs, reconstruct_physical_states

class ExtendedKalmanFilter:
    def __init__(
        self,
        Phi: np.ndarray,          # (p,p)
        Q: np.ndarray,            # (p,p)
        A: Callable,              # measurement function
        J: Callable,              # Jacobian of A
        s_0: np.ndarray,          # (p,)
        P_0: np.ndarray,          # (p,p)
        obs: np.ndarray,          # (n,q)
        meas_var: np.ndarray,     # (n,q)
        jitter: float = 1e-8,
        use_pinv: bool = False,
        state_min: Optional[np.ndarray] = None,
        state_max: Optional[np.ndarray] = None,
        standardize: bool = False,
    ):
        # -----------------------------------------------------
        # dimensions
        # -----------------------------------------------------
        self.p = s_0.shape[0]
        self.n = obs.shape[0]
        self.q = obs.shape[1]

        # -----------------------------------------------------
        # model
        # -----------------------------------------------------
        self.Phi = Phi
        self.Q = Q
        self.A = A
        self.J = J

        self.s_0 = s_0.reshape(self.p)
        self.P_0 = P_0

        # -----------------------------------------------------
        # observations
        # -----------------------------------------------------
        
        self.obs = obs.reshape(self.n, self.q)

        # measurement noise covariance R_t (n,q,q)
        self.R = np.zeros((self.n, self.q, self.q))
        for t in range(self.n):
            self.R[t] = np.diag(meas_var[t])

        self.jitter = jitter
        self.use_pinv = use_pinv

        self._check_dimensions()

        # -----------------------------------------------------
        # filter state
        # -----------------------------------------------------
        self.s_pred = None
        self.p_pred = None
        self.s_upd = None
        self.p_upd = None
        self.K = None

        # history (initialized lazily in filter())
        self.s_pred_hist = None
        self.p_pred_hist = None
        self.s_upd_hist = None
        self.p_upd_hist = None
        self.gains_hist = None

        # smoother
        self.smooth_s_hist = None
        self.smooth_p_hist = None
        self.smooth_gains_hist = None
        self.lag_one_cov_hist = None

        # constraints
        self.state_min = state_min
        self.state_max = state_max

        # bookkeeping
        self._filtered = False
        self._smoothed = False

    # ---------------------------------------------------------
    # dimension checks
    # ---------------------------------------------------------
    def _check_dimensions(self):
        assert self.Phi.shape == (self.p, self.p)
        assert self.Q.shape == (self.p, self.p)
        assert self.P_0.shape == (self.p, self.p)
        assert self.obs.shape == (self.n, self.q)
        assert self.R.shape == (self.n, self.q, self.q)

    # ---------------------------------------------------------
    # constraints
    # ---------------------------------------------------------
    def _apply_state_constraints_old(self, s: np.ndarray) -> np.ndarray:
        s_clamped = s.copy()
        if self.state_min is not None:
            s_clamped = np.maximum(s_clamped, self.state_min)
        if self.state_max is not None:
            s_clamped = np.minimum(s_clamped, self.state_max)
        return s_clamped
    
    def _apply_state_constraints(self, s: np.ndarray) -> np.ndarray:
        s_clamped = s.copy()

        if self.state_min is not None:
            for i in range(len(s_clamped)):
                if self.state_min[i] is not None:
                    s_clamped[i] = max(s_clamped[i], self.state_min[i])

        if self.state_max is not None:
            for i in range(len(s_clamped)):
                if self.state_max[i] is not None:
                    s_clamped[i] = min(s_clamped[i], self.state_max[i])

        return s_clamped

    # ---------------------------------------------------------
    # prediction
    # ---------------------------------------------------------
    def predict(self):
        I = np.eye(self.p)

        if self.s_upd is None:
            self.s_pred = self.Phi @ self.s_0
            self.s_pred = self._apply_state_constraints(self.s_pred)
            self.p_pred = self.Phi @ self.P_0 @ self.Phi.T + self.Q
        else:
            self.s_pred = self.Phi @ self.s_upd
            self.p_pred = self.Phi @ self.p_upd @ self.Phi.T + self.Q

        # numerical stability
        self.p_pred = 0.5 * (self.p_pred + self.p_pred.T)
        self.p_pred += self.jitter * I

    # ---------------------------------------------------------
    # update
    # ---------------------------------------------------------
    def update(self, x: np.ndarray, R: np.ndarray):
        I = np.eye(self.p)

        J = self.J(self.s_pred)
        x_pred = self.A(self.s_pred).reshape(self.q)

        S = J @ self.p_pred @ J.T + R
        S = 0.5 * (S + S.T)
        S += self.jitter * np.eye(self.q)

        if self.use_pinv:
            S_inv = np.linalg.pinv(S)
        else:
            S_inv = np.linalg.inv(S)

        self.K = self.p_pred @ J.T @ S_inv

        innovation = x - x_pred

        self.s_upd = self.s_pred + self.K @ innovation
        self.s_upd = self._apply_state_constraints(self.s_upd)

        # Joseph form (stable)
        IKH = I - self.K @ J
        self.p_upd = IKH @ self.p_pred @ IKH.T + self.K @ R @ self.K.T

        self.p_upd = 0.5 * (self.p_upd + self.p_upd.T)
        self.p_upd += self.jitter * I

    # ---------------------------------------------------------
    # one step
    # ---------------------------------------------------------
    def step(self, x: np.ndarray, R: np.ndarray):
        self.predict()
        self.update(x, R)

    # ---------------------------------------------------------
    # full filter
    # ---------------------------------------------------------
    def filter(self):
        self.s_upd = None
        self.p_upd = None

        # allocate histories
        self.s_pred_hist = np.zeros((self.n, self.p))
        self.p_pred_hist = np.zeros((self.n, self.p, self.p))
        self.s_upd_hist = np.zeros((self.n, self.p))
        self.p_upd_hist = np.zeros((self.n, self.p, self.p))
        self.gains_hist = np.zeros((self.n, self.p, self.q))

        for t in range(self.n):
            self.step(self.obs[t], self.R[t])
            self.s_pred_hist[t] = self.s_pred
            self.p_pred_hist[t] = self.p_pred
            self.s_upd_hist[t] = self.s_upd
            self.p_upd_hist[t] = self.p_upd
            self.gains_hist[t] = self.K

        self._filtered = True

        return (
            self.s_pred_hist,
            self.p_pred_hist,
            self.s_upd_hist,
            self.p_upd_hist,
            self.gains_hist,
        )

    # ---------------------------------------------------------
    # RTS smoother
    # ---------------------------------------------------------
    def smooth(self):
        if not self._filtered:
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

            P_pred_next = 0.5 * (P_pred_next + P_pred_next.T)
            P_pred_next += self.jitter * I

            if self.use_pinv:
                inv_Ppred = np.linalg.pinv(P_pred_next)
            else:
                inv_Ppred = np.linalg.inv(P_pred_next)

            J = P_upd @ self.Phi.T @ inv_Ppred
            J_smooth[t] = J

            s_smooth[t] = (
                self.s_upd_hist[t]
                + J @ (s_smooth[t + 1] - self.s_pred_hist[t + 1])
            )

            P_smooth[t] = (
                P_upd
                + J @ (P_smooth[t + 1] - P_pred_next) @ J.T
            )

            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
            P_smooth[t] += self.jitter * I

            lag_one_cov[t] = J @ P_smooth[t + 1]

        for t in range(n):
            s_smooth[t] = self._apply_state_constraints(s_smooth[t])

        self.smooth_s_hist = s_smooth
        self.smooth_p_hist = P_smooth
        self.smooth_gains_hist = J_smooth
        self.lag_one_cov_hist = lag_one_cov

        self._smoothed = True

        return s_smooth, P_smooth, J_smooth, lag_one_cov

    # ---------------------------------------------------------
    # full simulation smoother (disturbances + states)
    # ---------------------------------------------------------
    def simulation_smoothing(self, seed: Optional[int] = None):
        """
        Approximate Durbin-Koopman simulation smoother.

        Returns
        -------
        eps_sim : (n,q)
            Simulated observation disturbances
        eta_sim : (n-1,p)
            Simulated state disturbances
        alpha_sim : (n,p)
            Simulated states
        y_sim : (n,q)
            Simulated observations
        """
        if seed is not None:
            np.random.seed(seed)

        if not self._filtered:
            self.filter()

        if not self._smoothed:
            self.smooth()

        n, p, q = self.n, self.p, self.q

        # -----------------------------------------------------
        # Allocate outputs
        # -----------------------------------------------------
        eps_sim = np.zeros((n, q))
        eta_sim = np.zeros((n - 1, p))
        alpha_sim = np.zeros((n, p))

        # -----------------------------------------------------
        # Backward recursion arrays
        # -----------------------------------------------------
        r_tilde = np.zeros((n + 1, p))
        N_tilde = np.zeros((n + 1, p, p))

        # -----------------------------------------------------
        # Precompute quantities
        # -----------------------------------------------------
        v = np.zeros((n, q))
        F = np.zeros((n, q, q))
        K = self.gains_hist.copy()
        L = np.zeros((n, p, p))

        for t in range(n):

            J_t = self.J(self.s_pred_hist[t])

            y_pred = self.A(self.s_pred_hist[t]).reshape(q)

            v[t] = self.obs[t] - y_pred

            F[t] = (
                J_t
                @ self.p_pred_hist[t]
                @ J_t.T
                + self.R[t]
            )

            F[t] = 0.5 * (F[t] + F[t].T)
            F[t] += self.jitter * np.eye(q)

            L[t] = self.Phi - K[t] @ J_t

        # -----------------------------------------------------
        # Backward disturbance recursion
        # -----------------------------------------------------
        for t in reversed(range(n)):

            if self.use_pinv:
                F_inv = np.linalg.pinv(F[t])
            else:
                F_inv = np.linalg.inv(F[t])

            J_t = self.J(self.s_pred_hist[t])
            H_t = self.R[t]

            W_t = (
                F_inv @ J_t
                - K[t].T @ N_tilde[t + 1] @ L[t]
            ).T @ H_t

            D_t = (
                F_inv
                + K[t].T @ N_tilde[t + 1] @ K[t]
            )

            C_t = H_t - H_t @ D_t @ H_t

            # numerical stabilization
            C_t = 0.5 * (C_t + C_t.T)
            C_t += self.jitter * np.eye(q)

            eigvals, eigvecs = np.linalg.eigh(C_t)
            eigvals = np.maximum(eigvals, 0.0)
            C_t = eigvecs @ np.diag(eigvals) @ eigvecs.T

            d_t = np.random.multivariate_normal(
                mean=np.zeros(q),
                cov=C_t,
            )

            if self.use_pinv:
                C_inv_d = np.linalg.pinv(C_t) @ d_t
            else:
                C_inv_d = np.linalg.pinv(C_t) @ d_t

            r_tilde[t] = (
                J_t.T @ F_inv @ v[t]
                - W_t @ C_inv_d
                + L[t].T @ r_tilde[t + 1]
            )

            if self.use_pinv:
                C_inv_WT = np.linalg.pinv(C_t) @ W_t.T
            else:
                C_inv_WT = np.linalg.pinv(C_t) @ W_t.T

            N_tilde[t] = (
                J_t.T @ F_inv @ J_t
                + W_t @ C_inv_WT
                + L[t].T @ N_tilde[t + 1] @ L[t]
            )

            N_tilde[t] = 0.5 * (
                N_tilde[t] + N_tilde[t].T
            )

            eps_sim[t] = (
                d_t
                + H_t
                @ (
                    F_inv @ v[t]
                    - K[t].T @ r_tilde[t + 1]
                )
            )

        # -----------------------------------------------------
        # Simulate state disturbances
        # -----------------------------------------------------
        for t in range(n - 1):

            C_eta = (
                self.Q
                - self.Q @ N_tilde[t + 1] @ self.Q
            )

            C_eta = 0.5 * (C_eta + C_eta.T)
            C_eta += self.jitter * np.eye(p)

            eigvals, eigvecs = np.linalg.eigh(C_eta)
            eigvals = np.maximum(eigvals, 0.0)
            C_eta = eigvecs @ np.diag(eigvals) @ eigvecs.T

            d_t = np.random.multivariate_normal(
                mean=np.zeros(p),
                cov=C_eta,
            )

            eta_sim[t] = (
                d_t
                + self.Q @ r_tilde[t + 1]
            )

        # -----------------------------------------------------
        # Simulate states
        # -----------------------------------------------------
        P0 = 0.5 * (self.P_0 + self.P_0.T)
        P0 += self.jitter * np.eye(p)

        alpha_sim[0] = (
            self.s_0
            + np.random.multivariate_normal(
                mean=np.zeros(p),
                cov=P0,
            )
        )

        for t in range(n - 1):

            alpha_sim[t + 1] = (
                self.Phi @ alpha_sim[t]
                + eta_sim[t]
            )

        # -----------------------------------------------------
        # Simulate observations
        # -----------------------------------------------------
        y_sim = np.zeros((n, q))

        for t in range(n):

            y_sim[t] = (
                self.A(alpha_sim[t]).reshape(q)
                + eps_sim[t]
            )

        return (
            eps_sim,
            eta_sim,
            alpha_sim,
            y_sim,
        )

    # ---------------------------------------------------------
    # data likelihood
    # ---------------------------------------------------------
    def data_likelihood(self) -> float:
        if not self._filtered:
            self.filter()

        ll = 0.0
        Iq = np.eye(self.q)

        for t in range(self.n):
            x_pred = self.s_pred_hist[t]          # (p,)
            P_pred = self.p_pred_hist[t]          # (p,p)
            R = self.R[t]                         # (q,q)

            h_pred = self.A(x_pred).reshape(self.q)   # (q,)
            J = self.J(x_pred)                        # (q,p)

            Sigma = J @ P_pred @ J.T + R              # (q,q)
            Sigma = 0.5 * (Sigma + Sigma.T)
            Sigma += self.jitter * Iq

            residual = self.obs[t] - h_pred           # (q,)

            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                raise np.linalg.LinAlgError(f"Non-PD innovation covariance at t={t}")

            if self.use_pinv:
                Sigma_inv = np.linalg.pinv(Sigma)
                quad = residual.T @ Sigma_inv @ residual
            else:
                quad = residual.T @ np.linalg.solve(Sigma, residual)

            ll += -0.5 * (logdet + quad + self.q * np.log(2 * np.pi))

        self.log_likelihood = float(ll)
        return self.log_likelihood
