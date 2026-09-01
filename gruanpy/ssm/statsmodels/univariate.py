import statsmodels.api as sm
import numpy as np

"""
Univariate Local Linear Trend Model
https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_local_linear_trend.html
"""

class UnivariateLLT(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Model order
        k_states = k_posdef = 2

        # Initialize the statespace
        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # Initialize the matrices
        self.ssm["design"] = np.array([1, 0])
        self.ssm["transition"] = np.array([[1, 1], [0, 1]])
        self.ssm["selection"] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["sigma2.measurement", "sigma2.level", "sigma2.trend"]

    @property
    def start_params(self):
        return [np.std(self.endog)] * 3

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)

        # Observation covariance
        self.ssm["obs_cov", 0, 0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]


"""
Univariate Local Linear Level Model (LLL)
y_t = level_t + eps_t
level_t = level_{t-1} + eta_t
"""
class UnivariateLLL(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        k_states = k_posdef = 1  # only one state: the level

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # Observation equation: y_t = [1] * level_t
        self.ssm["design"] = np.array([1.0])

        # State transition: level_t = level_{t-1} + eta_t
        self.ssm["transition"] = np.array([[1.0]])

        # Selection matrix (noise enters the state)
        self.ssm["selection"] = np.eye(k_states)

        # Cache diagonal indices for state covariance
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["sigma2.measurement", "sigma2.level"]

    @property
    def start_params(self):
        # reasonable initial guesses
        return [np.std(self.endog), np.std(self.endog)]

    def transform_params(self, unconstrained):
        # enforce positivity
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)

        # Measurement noise variance
        self.ssm["obs_cov", 0, 0] = params[0]

        # State noise variance (level innovation)
        self.ssm[self._state_cov_idx] = params[1]