import numpy as np
from typing import Tuple

from ssm.from_scratch.ekf import ExtendedKalmanFilter


class EKF_EM:
    def __init__(self, ekf: ExtendedKalmanFilter):
        self.ekf = ekf

        self.S11 = None
        self.S10 = None
        self.S00 = None

        self.EM_log_likelihoods = []
        self.EM_Q = []
        self.EM_s0 = []
        self.EM_P0 = []

    # ---------------------------------------------------------
    # S-matrices
    # ---------------------------------------------------------
    def _compute_S_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.ekf._smoothed:
            raise RuntimeError("Run E_step() (filter + smooth) before computing S matrices.")

        p = self.ekf.p
        n = self.ekf.n

        S11 = np.zeros((p, p))
        S10 = np.zeros((p, p))
        S00 = np.zeros((p, p))

        for t in range(n):
            x_t = self.ekf.smooth_s_hist[t]       # (p,)
            P_t = self.ekf.smooth_p_hist[t]       # (p,p)

            S11 += P_t + np.outer(x_t, x_t)

            if t > 0:
                P_t_tm1 = self.ekf.lag_one_cov_hist[t - 1]   # (p,p)
                x_tm1 = self.ekf.smooth_s_hist[t - 1]
                S10 += P_t_tm1 + np.outer(x_t, x_tm1)

            if t < n - 1:
                S00 += P_t + np.outer(x_t, x_t)

        self.S11 = S11
        self.S10 = S10
        self.S00 = S00

        return S11, S10, S00

    # ---------------------------------------------------------
    # E-step
    # ---------------------------------------------------------
    def E_step(self):
        self.ekf.filter()
        self.ekf.smooth()
        self._compute_S_matrices()

    # ---------------------------------------------------------
    # M-step
    # ---------------------------------------------------------
    def M_step(self):
        n = self.ekf.n
        Phi = self.ekf.Phi

        if self.S11 is None or self.S10 is None or self.S00 is None:
            raise RuntimeError("Run E_step() before M_step().")

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

        self.ekf.Q = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # --- update initial state ---
        self.ekf.s_0 = self.ekf.smooth_s_hist[0].copy()

        # --- update initial covariance ---
        P0 = 0.5 * (self.ekf.smooth_p_hist[0] + self.ekf.smooth_p_hist[0].T)
        eigvals, eigvecs = np.linalg.eigh(P0)
        eigvals = np.clip(eigvals, 1e-10, None)
        self.ekf.P_0 = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # ---------------------------------------------------------
    # EM algorithm
    # ---------------------------------------------------------
    def EM_algorithm(self, max_iter: int = 100, tol: float = 1e-6, verbose: bool = False):
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
            self.EM_Q.append(self.ekf.Q.copy())
            self.EM_s0.append(self.ekf.s_0.copy())
            self.EM_P0.append(self.ekf.P_0.copy())

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
