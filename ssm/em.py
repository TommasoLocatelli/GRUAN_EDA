# em.py
import numpy as np
from typing import Tuple, List
from ekf import EKF
from model import Model

class EMEstimator:
    """
    EM estimator for linear dynamics with nonlinear measurements using EKF smoothing.
    It accepts an EKF instance (with its Model) and runs EM to update Q, s_0, P_0.
    """

    def __init__(self, ekf: EKF):
        self.ekf = ekf
        self.model = ekf.model
        self.n = ekf.n
        self.p = ekf.p

        # S matrices placeholders
        self.S11 = None
        self.S10 = None
        self.S00 = None

        # history
        self.EM_log_likelihoods: List[float] = []
        self.EM_Q: List[np.ndarray] = []
        self.EM_s0: List[np.ndarray] = []
        self.EM_P0: List[np.ndarray] = []

    # -------------------------
    # E-step: filter + smooth + compute S
    # -------------------------
    def E_step(self):
        self.ekf.filter()
        self.ekf.smooth()
        self._compute_S_matrices()

    def _compute_S_matrices(self):
        p = self.p
        n = self.n

        S11 = np.zeros((p, p))
        S10 = np.zeros((p, p))
        S00 = np.zeros((p, p))

        for t in range(n):
            x_t = self.ekf.smooth_s_hist[t]
            P_t = self.ekf.smooth_p_hist[t]
            S11 += P_t + np.outer(x_t, x_t)

            if t > 0:
                P_t_tm1 = self.ekf.lag_one_cov_hist[t - 1]
                x_tm1 = self.ekf.smooth_s_hist[t - 1]
                S10 += P_t_tm1 + np.outer(x_t, x_tm1)

            if t < n - 1:
                S00 += P_t + np.outer(x_t, x_t)

        self.S11 = S11
        self.S10 = S10
        self.S00 = S00

        return S11, S10, S00

    # -------------------------
    # M-step: update Q, s_0, P_0 (in-place on model)
    # -------------------------
    def M_step(self):
        n = self.n
        Phi = self.model.Phi

        Q_new = (
            self.S11
            - Phi @ self.S10.T
            - self.S10 @ Phi.T
            + Phi @ self.S00 @ Phi.T
        ) / max(1, (n - 1))

        Q_new = 0.5 * (Q_new + Q_new.T)
        eigvals, eigvecs = np.linalg.eigh(Q_new)
        eigvals = np.clip(eigvals, 1e-10, None)
        Q_new = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # update model in-place
        self.model.Q = Q_new

        # update initial state
        self.model.s_0 = self.ekf.smooth_s_hist[0].copy()

        # update initial covariance
        P0 = 0.5 * (self.ekf.smooth_p_hist[0] + self.ekf.smooth_p_hist[0].T)
        eigvals, eigvecs = np.linalg.eigh(P0)
        eigvals = np.clip(eigvals, 1e-10, None)
        self.model.P_0 = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # after updating model, reset EKF internals so next E-step uses new model
        self.ekf.reset()

    # -------------------------
    # EM loop
    # -------------------------
    def run(self, max_iter: int = 100, tol: float = 1e-6, verbose: bool = False) -> Tuple[List[float], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        prev_ll = -np.inf
        self.EM_log_likelihoods = []
        self.EM_Q = []
        self.EM_s0 = []
        self.EM_P0 = []

        for k in range(max_iter):
            self.E_step()
            self.M_step()

            ll = self.ekf.data_likelihood()

            self.EM_log_likelihoods.append(ll)
            self.EM_Q.append(self.model.Q.copy())
            self.EM_s0.append(self.model.s_0.copy())
            self.EM_P0.append(self.model.P_0.copy())

            if verbose:
                print(f"EM iter {k+1}: logLik = {ll:.6f}")

            if abs(ll - prev_ll) < tol:
                if verbose:
                    print(f"Converged after {k+1} iterations.")
                break

            prev_ll = ll

        return (self.EM_log_likelihoods, self.EM_Q, self.EM_s0, self.EM_P0)
