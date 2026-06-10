import numpy as np
from typing import Callable
import warnings


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
        state_min: np.ndarray | None = None,
        state_max: np.ndarray | None = None,
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

        # history
        self.s_pred_hist = np.zeros((self.n, self.p))
        self.p_pred_hist = np.zeros((self.n, self.p, self.p))
        self.s_upd_hist = np.zeros((self.n, self.p))
        self.p_upd_hist = np.zeros((self.n, self.p, self.p))
        self.gains_hist = np.zeros((self.n, self.p, self.q))

        # smoother
        self.smooth_s_hist = None
        self.smooth_p_hist = None
        self.smooth_gains_hist = None
        self.lag_one_cov_hist = None

        self.jitter = jitter
        self.use_pinv = use_pinv

        # vincoli fisici sugli stati
        self.state_min = state_min
        self.state_max = state_max


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
    def _apply_state_constraints(self, s: np.ndarray) -> np.ndarray:
        s_clamped = s.copy()
        if self.state_min is not None:
            s_clamped = np.maximum(s_clamped, self.state_min)
        if self.state_max is not None:
            s_clamped = np.minimum(s_clamped, self.state_max)
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

        # stabilità numerica
        self.p_pred = 0.5 * (self.p_pred + self.p_pred.T)
        self.p_pred += self.jitter * I

    # ---------------------------------------------------------
    # update
    # ---------------------------------------------------------
    def update(self, x, R):
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


        # Joseph form (stabile)
        IKH = I - self.K @ J
        self.p_upd = IKH @ self.p_pred @ IKH.T + self.K @ R @ self.K.T

        self.p_upd = 0.5 * (self.p_upd + self.p_upd.T)
        self.p_upd += self.jitter * I

    # ---------------------------------------------------------
    # one step
    # ---------------------------------------------------------
    def step(self, x, R):
        self.predict()
        self.update(x, R)

    # ---------------------------------------------------------
    # full filter
    # ---------------------------------------------------------
    def filter(self):
        self.s_upd = None
        self.p_upd = None

        for t in range(self.n):
            self.step(self.obs[t], self.R[t])
            self.s_pred_hist[t] = self.s_pred
            self.p_pred_hist[t] = self.p_pred
            self.s_upd_hist[t] = self.s_upd
            self.p_upd_hist[t] = self.p_upd
            self.gains_hist[t] = self.K

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
        if self.s_pred_hist is None:
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

        return s_smooth, P_smooth, J_smooth, lag_one_cov

    # ---------------------------------------------------------
    # simulation smoother
    # ---------------------------------------------------------
    def simulate_states(self):
        if self.smooth_s_hist is None:
            raise RuntimeError("Run smooth() first.")

        n, p = self.n, self.p
        states = np.zeros((n, p))

        states[-1] = np.random.multivariate_normal(
            self.smooth_s_hist[-1],
            self.smooth_p_hist[-1]
        )

        for t in range(n - 2, -1, -1):
            P_t = self.smooth_p_hist[t]
            P_tp1 = self.smooth_p_hist[t + 1]
            P_pred_next = self.p_pred_hist[t + 1]

            J = P_t @ self.Phi.T @ np.linalg.inv(P_pred_next)

            mean = self.smooth_s_hist[t] + J @ (states[t + 1] - self.smooth_s_hist[t + 1])
            cov = P_t - J @ P_tp1 @ J.T

            states[t] = np.random.multivariate_normal(mean, cov)

        return states

    # ---------------------------------------------------------
    # simulate observations
    # ---------------------------------------------------------
    def simulate_observations(self, states):
        n, q = self.n, self.q
        obs_sim = np.zeros((n, q))

        for t in range(n):
            mean = self.A(states[t])
            noise = np.random.multivariate_normal(np.zeros(q), self.R[t])
            obs_sim[t] = mean + noise

        return obs_sim
        

    def data_likelihood(self):

        if self.s_pred_hist is None:
            self.filter()

        ll = 0.0

        for t in range(self.n):

            x_pred = self.s_pred_hist[t]          # (p,)
            P_pred = self.p_pred_hist[t]          # (p,p)
            R = self.R[t]                         # (q,q)

            h_pred = self.A(x_pred).reshape(self.q)   # (q,)
            J = self.J(x_pred)                        # (q,p)

            Sigma = J @ P_pred @ J.T + R              # (q,q)
            residual = self.obs[t] - h_pred           # (q,)

            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                raise np.linalg.LinAlgError(f"Non-PD innovation covariance at t={t}")

            quad = residual.T @ np.linalg.solve(Sigma, residual)

            ll += -0.5 * (logdet + quad + self.q * np.log(2*np.pi))

        self.log_likelihood = float(ll)
        return self.log_likelihood

    def _compute_S_matrices(self):

        p = self.p
        n = self.n

        S11 = np.zeros((p, p))
        S10 = np.zeros((p, p))
        S00 = np.zeros((p, p))

        for t in range(n):

            x_t = self.smooth_s_hist[t]       # (p,)
            P_t = self.smooth_p_hist[t]       # (p,p)

            S11 += P_t + np.outer(x_t, x_t)

            if t > 0:
                P_t_tm1 = self.lag_one_cov_hist[t - 1]   # (p,p)
                x_tm1 = self.smooth_s_hist[t - 1]
                S10 += P_t_tm1 + np.outer(x_t, x_tm1)

            if t < n - 1:
                S00 += P_t + np.outer(x_t, x_t)

        self.S11 = S11
        self.S10 = S10
        self.S00 = S00

        return S11, S10, S00

    def E_step(self):
        self.filter()
        self.smooth()
        self._compute_S_matrices()

    def M_step(self):

        n = self.n
        Phi = self.Phi

        # --- update Q ---
        Q_new = (
            self.S11
            - Phi @ self.S10.T
            - self.S10 @ Phi.T
            + Phi @ self.S00 @ Phi.T
        ) / (n - 1)

        Q_new = 0.5 * (Q_new + Q_new.T)

        eigvals, eigvecs = np.linalg.eigh(Q_new)
        eigvals = np.clip(eigvals, 1e-10, None)

        self.Q = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # --- update initial state ---
        self.s_0 = self.smooth_s_hist[0].copy()

        # --- update initial covariance ---
        P0 = 0.5 * (self.smooth_p_hist[0] + self.smooth_p_hist[0].T)
        eigvals, eigvecs = np.linalg.eigh(P0)
        eigvals = np.clip(eigvals, 1e-10, None)
        self.P_0 = eigvecs @ np.diag(eigvals) @ eigvecs.T

    def EM_algorithm(self, max_iter=100, tol=1e-6, verbose=False):

        prev_ll = -np.inf

        self.EM_log_likelihoods = []
        self.EM_Q = []
        self.EM_s0 = []
        self.EM_P0 = []

        for k in range(max_iter):

            self.E_step()
            self.M_step()

            ll = self.data_likelihood()

            self.EM_log_likelihoods.append(ll)
            self.EM_Q.append(self.Q.copy())
            self.EM_s0.append(self.s_0.copy())
            self.EM_P0.append(self.P_0.copy())

            if verbose:
                print(f"EM iter {k+1}: logLik = {ll:.6f}")

            if abs(ll - prev_ll) < tol:
                if verbose:
                    print(f"Converged after {k+1} iterations.")
                break

            prev_ll = ll

        return (
            self.EM_log_likelihoods,
            self.EM_Q,
            self.EM_s0,
            self.EM_P0,
        )
