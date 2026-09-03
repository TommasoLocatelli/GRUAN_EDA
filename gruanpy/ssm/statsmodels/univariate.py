import statsmodels.api as sm
import numpy as np

"""
Univariate Local Linear Trend Model
https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_local_linear_trend.html

Parameters can also be fixed 
results = model.fit_constrained({'sigma2.measurement': 0})
"""

class UnivariateLLT(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, measurement_sigma2=None):
        k_states = k_posdef = 2

        # Store fixed measurement variance sequence if provided
        self.measurement_sigma2 = (
            measurement_sigma2.T if measurement_sigma2 is not None else None
        )

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        self.ssm["design"] = np.array([1, 0])
        self.ssm["transition"] = np.array([[1, 1], [0, 1]])
        self.ssm["selection"] = np.eye(k_states)

        # Univariate obs_cov
        self.ssm["obs_cov"] = np.zeros((1, 1, self.nobs))

        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        if self.measurement_sigma2 is None:
            return ["sigma2.measurement", "sigma2.level", "sigma2.trend"]
        else:
            return ["sigma2.level", "sigma2.trend"]

    @property
    def start_params(self):
        if self.measurement_sigma2 is None:
            return [np.std(self.endog)] * 3
        else:
            return [np.std(self.endog)] * 2

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)

        # Observation covariance
        if self.measurement_sigma2 is None:
            self.ssm["obs_cov", 0, 0] = params[0]
            state_params = params[1:]
        else:
            self.ssm["obs_cov", 0, 0] = self.measurement_sigma2[:]
            state_params = params

        # State covariance
        self.ssm[self._state_cov_idx] = state_params


"""
Univariate Local Linear Level Model (LLL)
y_t = level_t + eps_t
level_t = level_{t-1} + eta_t
"""
class UnivariateLLL(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, measurement_sigma2=None):
        k_states = k_posdef = 1  # only one state: the level

        # Store fixed measurement variance sequence if provided
        self.measurement_sigma2 = (
            measurement_sigma2.T if measurement_sigma2 is not None else None
        )

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

        # Univariate obs_cov (time-varying if measurement_sigma2 is provided)
        self.ssm["obs_cov"] = np.zeros((1, 1, self.nobs))

        # Cache diagonal indices for state covariance
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        if self.measurement_sigma2 is None:
            return ["sigma2.measurement", "sigma2.level"]
        else:
            return ["sigma2.level"]

    @property
    def start_params(self):
        if self.measurement_sigma2 is None:
            return [np.std(self.endog), np.std(self.endog)]
        else:
            return [np.std(self.endog)]

    def transform_params(self, unconstrained):
        # enforce positivity
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)

        # Measurement noise variance
        if self.measurement_sigma2 is None:
            # scalar parameter
            self.ssm["obs_cov", 0, 0] = params[0]
            state_params = params[1:]
        else:
            # fixed time-varying measurement variance
            self.ssm["obs_cov", 0, 0] = self.measurement_sigma2[:]
            state_params = params

        # State noise variance (level innovation)
        self.ssm[self._state_cov_idx] = state_params
