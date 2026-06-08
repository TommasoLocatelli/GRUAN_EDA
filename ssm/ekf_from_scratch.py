import numpy as np
from typing import Callable
import warnings

"""
Extended Kalman Filter implementation.
Assumes a nonlinear measurement function A(s) and linear state transition with matrix Phi.
The Jacobian of the measurement function is provided by the user via the J argument.
The filter includes a Rauch-Tung-Striebel (RTS) smoother with lag-one covariance computation, as well as an EM algorithm for learning the process noise covariance Q and initial state parameters.
"""

class ExtendedKalmanFilter:

    def __init__(
        self,
        Phi: np.ndarray, # state transition matrix
        Q: np.ndarray, # process noise covariance
        A: Callable, # measurement function, eventually nonlinear
        J: Callable, # function computing Jacobian of measurement function from state
        s_0: np.ndarray, # initial state estimate
        P_0: np.ndarray, # initial error covariance
        obs: list[np.ndarray], # list of observation vectors
        meas_var: list[np.ndarray], # list of measurement noise variances
    ):

        # =====================================================
        # dimensions
        # =====================================================

        self.p = s_0.shape[0] # state dimension
        self.n = len(obs) # number of time steps
        self.q = obs[0].shape[0] # observation dimension

        # =====================================================
        # model parameters
        # =====================================================

        self.Phi = Phi
        self.Q = Q
        self.A = A
        self.J = J

        self.s_0 = s_0
        self.P_0 = P_0

        # =====================================================
        # observations
        # =====================================================

        self.obs = obs

        # list of measurement noise covariance matrices (diagonal, constructed from meas_var)
        self.R = []
        for var in meas_var:
            arr = np.asarray(var)
            arr = arr.flatten()
            if len(arr) != self.q:
                raise ValueError(
                    "Measurement variance dimension mismatch."
                )
            R = np.diag(arr)
            self.R.append(R)

        self.dimensions_check()

        # =====================================================
        # current filter state
        # =====================================================

        self.s_pred = None # predicted state
        self.p_pred = None # predicted covariance
        self.s_upd = None # updated/filtered state
        self.p_upd = None # updated/filtered covariance
        self.K = None # Kalman gain

        # =====================================================
        # filter history, one entry per time step
        # =====================================================

        self.s_pred_hist = None # list of predicted state
        self.p_pred_hist = None # list of predicted covariance
        self.s_upd_hist = None # list of updated/filtered state
        self.p_upd_hist = None # list of updated/filtered covariance
        self.gains_hist = None # list of Kalman gain

        # =====================================================
        # current smoothed estimates
        # =====================================================

        self.smooth_s = None # smoothed state
        self.smooth_p = None # smoothed covariance
        self.smooth_gains = None # RTS smoother gain
        self.lag_one_cov = None # P_{t,t-1|n}

        # =====================================================
        # smoother history, one entry per time step
        # =====================================================
        
        self.smooth_s_hist = None # list of smoothed states
        self.smooth_p_hist = None # list of smoothed covariances
        self.smooth_gains_hist = None # list of RTS gains
        self.lag_one_cov_hist = None # list of lag-one covariances

        # =====================================================
        # current EM algorithm
        # =====================================================

        self.S11 = None
        self.S10 = None
        self.S00 = None
        self.log_likelihood = None

        # =====================================================
        # EM algorithm history
        # =====================================================

        self.log_likelihood_hist = []
        self.EM_Q = []
        self.EM_s0 = []
        self.EM_P0 = []

    # =========================================================
    # dimension checks
    # =========================================================

    def dimensions_check(self):

        assert self.Phi.shape == (self.p, self.p)

        assert self.Q.shape == (self.p, self.p)

        assert self.s_0.shape == (self.p, 1)

        assert self.P_0.shape == (self.p, self.p)

        for i in range(self.n):

            assert self.obs[i].shape == (self.q, 1)

            assert self.R[i].shape == (self.q, self.q)

    # =========================================================
    # Kalman filter
    # =========================================================

    def predict(self):

        if self.s_upd is None: # first time step, use initial state
            self.s_pred = self.Phi @ self.s_0 # state predicition
            self.p_pred = ( # covariance prediction
                self.Phi
                @ self.P_0
                @ self.Phi.T
                + self.Q
            )

        else: # subsequent time steps, use previous updated state
            self.s_pred = self.Phi @ self.s_upd # state prediction
            self.p_pred = ( # covariance prediction
                self.Phi
                @ self.p_upd
                @ self.Phi.T
                + self.Q
            )

    def update(self, x, R):

        J = self.J(self.s_pred) # Jacobian of measurement function at predicted state
        x_pred = np.asarray(self.A(self.s_pred)).reshape(-1, 1) # predicted measurement at predicted state

        # innovation covariance
        S = (
            J
            @ self.p_pred
            @ J.T
            + R
        )

        # Kalman gain
        self.K = np.linalg.solve(
            S.T,
            (J @ self.p_pred.T)
        ).T

        innovation = x - x_pred

        # state update
        self.s_upd = (
            self.s_pred
            + self.K @ innovation
        )

        # guards against negative estimates
        for i in range(self.s_upd.shape[0]):
            if self.s_upd[i] < 0:
                self.s_upd[i] = 1e-3
                warnings.warn(
                    f"Negative state estimate at index {i} corrected to 1e-3."
                )
        
        # Joseph stabilized covariance update
        I = np.eye(self.p)

        self.p_upd = (
            (I - self.K @ J)
            @ self.p_pred
            @ (I - self.K @ J).T
            + self.K @ R @ self.K.T
        )

        # enforce symmetry
        self.p_upd = 0.5 * (
            self.p_upd + self.p_upd.T
        )

    def step(self, x, R):

        self.predict()

        self.update(x, R)

    def filter(self):

        n = self.n
        p = self.p

        # reset
        self.s_upd = None
        self.p_upd = None

        self.s_pred_hist = np.zeros((n, p, 1))
        self.p_pred_hist = np.zeros((n, p, p))
        self.s_upd_hist = np.zeros((n, p, 1))
        self.p_upd_hist = np.zeros((n, p, p))
        self.gains_hist = np.zeros((n, p, self.q))

        for t in range(n):
            
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

    def _smooth_with_cross_cov(self):
        # RTS smoother + lag-one covariance

        n = self.n
        p = self.p
        self.smooth_s_hist = np.zeros((n, p, 1))
        self.smooth_p_hist = np.zeros((n, p, p))
        self.smooth_gains_hist = np.zeros((n - 1, p, p))

        # initialize
        self.smooth_s_hist[-1] = self.s_upd_hist[-1]
        self.smooth_p_hist[-1] = self.p_upd_hist[-1]

        # RTS backward pass
        for t in range(n - 2, -1, -1):

            P_upd = self.p_upd_hist[t]
            P_pred_next = self.p_pred_hist[t + 1]

            # RTS gain
            J = P_upd @ self.Phi.T @ np.linalg.solve(P_pred_next, np.eye(p))
            self.smooth_gains_hist[t] = J

            # smoothed state
            self.smooth_s_hist[t] = (
                self.s_upd_hist[t]
                + J @ (self.smooth_s_hist[t + 1] - self.s_pred_hist[t + 1])
            )

            # smoothed covariance
            self.smooth_p_hist[t] = (
                P_upd
                + J @ (self.smooth_p_hist[t + 1] - P_pred_next) @ J.T
            )

            # symmetrize
            self.smooth_p_hist[t] = 0.5 * (
                self.smooth_p_hist[t] + self.smooth_p_hist[t].T
            )

        # -----------------------------------------------------
        # lag-one covariance
        # -----------------------------------------------------
        P_cross = np.zeros((n - 1, p, p))

        # final lag-one covariance (t = n-1)
        K_last = self.gains_hist[-1]
        H_last = self.J(self.s_pred_hist[-1])

        P_cross[-1] = (
            (np.eye(p) - K_last @ H_last)
            @ self.Phi
            @ self.p_upd_hist[-2]
        )

        # recursive lag-one covariance
        for t in range(n - 3, -1, -1):
            J_t = self.smooth_gains_hist[t]
            J_tp1 = self.smooth_gains_hist[t + 1]
            P_cross[t] = (
                self.p_upd_hist[t + 1] @ J_t.T
                + J_tp1 @ (
                    P_cross[t + 1]
                    - self.Phi @ self.p_upd_hist[t + 1]
                ) @ J_t.T
            )
        self.lag_one_cov_hist = P_cross

        return self.smooth_s_hist, self.smooth_p_hist, self.lag_one_cov_hist

    def smooth(self):

        if self.s_pred_hist is None:
            self.filter()
        self._smooth_with_cross_cov()

        return (
            self.smooth_s_hist,
            self.smooth_p_hist,
            self.smooth_gains_hist,
            self.lag_one_cov_hist
        )

    # =========================================================
    # log-likelihood
    # =========================================================

    def data_likelihood(self):

        ll = 0.0

        for t in range(self.n):

            x_pred = self.filter_s_pred[t]      # (p,1)
            P_pred = self.filter_p_pred[t]      # (p,p)
            R = self.R[t]                       # (q,q)

            # -----------------------------------------------------
            # EKF measurement prediction
            # -----------------------------------------------------
            h_pred = np.asarray(self.A(x_pred)).reshape(-1, 1)   # (q,1)
            J = self.J(x_pred)                                   # (q,p)

            # -----------------------------------------------------
            # Innovation covariance
            # -----------------------------------------------------
            Sigma = J @ P_pred @ J.T + R                         # (q,q)

            # -----------------------------------------------------
            # Residual
            # -----------------------------------------------------
            residual = self.obs[t] - h_pred                      # (q,1)

            # -----------------------------------------------------
            # Log-likelihood contribution
            # -----------------------------------------------------
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                raise np.linalg.LinAlgError(
                    f"Non-positive definite innovation covariance at t={t}"
                )

            quad = float(residual.T @ np.linalg.solve(Sigma, residual))

            ll += -0.5 * (
                logdet
                + quad
                + self.q * np.log(2 * np.pi)
            )

        self.log_likelihood = float(ll)
        return self.log_likelihood


    # =========================================================
    # EM sufficient statistics
    # =========================================================

    def _compute_S_matrices(self):

        S11 = np.zeros((self.p, self.p))

        S10 = np.zeros((self.p, self.p))

        S00 = np.zeros((self.p, self.p))

        for t in range(self.n):

            x_t = self.smooth_s[t]

            P_t = self.smooth_p[t]

            S11 += (
                P_t
                + x_t @ x_t.T
            )

            if t > 0:

                P_t_tm1 = self.lag_one_cov[t - 1]

                x_tm1 = self.smooth_s[t - 1]

                S10 += (
                    P_t_tm1
                    + x_t @ x_tm1.T
                )

            if t < self.n - 1:

                S00 += (
                    P_t
                    + x_t @ x_t.T
                )

        self.S11 = S11
        self.S10 = S10
        self.S00 = S00

        return S11, S10, S00

    # =========================================================
    # EM steps
    # =========================================================

    def E_step(self):

        self.filter()

        self._smooth_with_cross_cov()

        self._compute_S_matrices()

    def M_step(self):

        n = self.n

        Phi = self.Phi

        # -----------------------------------------------------
        # update Q
        # -----------------------------------------------------

        Q_new = (
            self.S11
            - Phi @ self.S10.T
            - self.S10 @ Phi.T
            + Phi @ self.S00 @ Phi.T
        ) / (n - 1)

        # symmetrize
        Q_new = 0.5 * (
            Q_new + Q_new.T
        )

        # PSD projection
        eigvals, eigvecs = np.linalg.eigh(Q_new)

        eigvals = np.clip(
            eigvals,
            1e-10,
            None
        )

        self.Q = (
            eigvecs
            @ np.diag(eigvals)
            @ eigvecs.T
        )

        # -----------------------------------------------------
        # update initial state
        # -----------------------------------------------------

        self.s_0 = self.smooth_s[0]

        # -----------------------------------------------------
        # update initial covariance
        # -----------------------------------------------------

        P0 = 0.5 * (
            self.smooth_p[0]
            + self.smooth_p[0].T
        )

        eigvals, eigvecs = np.linalg.eigh(P0)

        eigvals = np.clip(
            eigvals,
            1e-10,
            None
        )

        self.P_0 = (
            eigvecs
            @ np.diag(eigvals)
            @ eigvecs.T
        )

    # =========================================================
    # EM algorithm
    # =========================================================

    def EM_algorithm(
        self,
        max_iter=100,
        tol=1e-6,
        verbose=False,
    ):

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

            self.EM_P0.append(self.p_0.copy())

            if verbose:

                print(
                    f"EM iter {k+1}: "
                    f"logLik = {ll:.6f}"
                )

            if abs(ll - prev_ll) < tol:

                if verbose:

                    print(
                        f"Converged after "
                        f"{k+1} iterations."
                    )

                break

            prev_ll = ll

        return (
            self.EM_log_likelihoods,
            self.EM_Q,
            self.EM_s0,
            self.EM_P0,
        )