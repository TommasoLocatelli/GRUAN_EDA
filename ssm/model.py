# model.py
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

@dataclass
class Model:
    Phi: np.ndarray           # (p,p)
    Q: np.ndarray             # (p,p)
    A: Callable               # measurement function: R^p -> R^q
    J: Callable               # Jacobian of A: R^p -> (q,p)
    s_0: np.ndarray           # (p,)
    P_0: np.ndarray           # (p,p)
    state_min: Optional[np.ndarray] = None
    state_max: Optional[np.ndarray] = None
    jitter: float = 1e-8
    use_pinv: bool = False

    def apply_state_constraints(self, s: np.ndarray) -> np.ndarray:
        s_clamped = s.copy()
        if self.state_min is not None:
            s_clamped = np.maximum(s_clamped, self.state_min)
        if self.state_max is not None:
            s_clamped = np.minimum(s_clamped, self.state_max)
        return s_clamped

    def ensure_pd(self, M: np.ndarray) -> np.ndarray:
        M = 0.5 * (M + M.T)
        if self.jitter > 0:
            M = M + self.jitter * np.eye(M.shape[0])
        return M
