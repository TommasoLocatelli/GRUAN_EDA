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

class DeterministicLevelLLT(sm.tsa.statespace.MLEModel):
    """
    Local Linear Trend model with deterministic level:
        level_t = level_{t-1} + trend_{t-1}
        trend_t = trend_{t-1} + η_t,   η_t ~ N(0, σ²_trend)
        y_t = level_t + ε_t,          ε_t ~ N(0, σ²_measurement)
    """

    def __init__(self, endog, measurement_sigma2=None):
        k_states = 2
        k_posdef = 2  # Q is 2x2 but level variance will be fixed at zero

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

        # State-space matrices
        self.ssm["design"] = np.array([1, 0])
        self.ssm["transition"] = np.array([[1, 1],
                                           [0, 1]])
        self.ssm["selection"] = np.eye(k_states)

        # Time-varying obs_cov
        self.ssm["obs_cov"] = np.zeros((1, 1, self.nobs))

        # Index for diagonal of state covariance
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    # Only trend variance is estimated
    @property
    def param_names(self):
        if self.measurement_sigma2 is None:
            return ["sigma2.measurement", "sigma2.trend"]
        else:
            return ["sigma2.trend"]

    @property
    def start_params(self):
        if self.measurement_sigma2 is None:
            return [np.var(self.endog), np.var(self.endog)]
        else:
            return [np.var(self.endog)]

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)

        # Observation variance
        if self.measurement_sigma2 is None:
            meas_var = params[0]
            trend_var = params[1]
        else:
            meas_var = self.measurement_sigma2[:]
            trend_var = params[0]

        self.ssm["obs_cov", 0, 0] = meas_var

        # State covariance Q:
        # level variance = 0 (deterministic)
        # trend variance = trend_var
        self.ssm["state_cov", 0, 0] = 0.0
        self.ssm["state_cov", 1, 1] = trend_var

        return params

class UnivariateLLT_AR1(sm.tsa.statespace.MLEModel):
    """
    Local Linear Trend + AR(1) measurement error.

    y_t = level_t + e_t

    level_t = level_{t-1} + trend_{t-1} + eta_t
    trend_t = trend_{t-1} + zeta_t
    e_t     = phi * e_{t-1} + u_t

    eta_t ~ N(0, sigma2_level)
    zeta_t ~ N(0, sigma2_trend)
    u_t ~ N(0, sigma2_error)
    """

    def __init__(self, endog, measurement_sigma2=None):

        k_states = 3
        k_posdef = 3

        self.measurement_sigma2 = (
            measurement_sigma2.T
            if measurement_sigma2 is not None
            else None
        )

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # y_t = level_t + error_t
        self.ssm["design"] = np.array([1.0, 0.0, 1.0])

        # State transition
        #
        # level_t = level_{t-1} + trend_{t-1}
        # trend_t = trend_{t-1}
        # error_t = phi * error_{t-1}
        #
        self.ssm["transition"] = np.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],   # phi inserted during update()
        ])

        self.ssm["selection"] = np.eye(k_states)

        # No additional white-noise measurement error.
        # The stochastic error is now a state.
        self.ssm["obs_cov"] = np.zeros((1, 1, self.nobs))

        self._state_cov_idx = (
            "state_cov",
            np.diag_indices(k_posdef)
        )

    @property
    def param_names(self):

        if self.measurement_sigma2 is None:
            return [
                "sigma2.level",
                "sigma2.trend",
                "sigma2.error",
                "phi",
            ]
        else:
            return [
                "sigma2.level",
                "sigma2.trend",
                "sigma2.error",
                "phi",
            ]

    @property
    def start_params(self):

        return np.array([
            np.var(self.endog) * 0.01,  # level
            np.var(self.endog) * 0.01,  # trend
            np.var(self.endog) * 0.5,   # AR error
            0.5,                       # phi
        ])

    def transform_params(self, unconstrained):

        params = unconstrained.copy()

        # Variances must be positive
        params[:3] = params[:3] ** 2

        # phi must be inside (-1, 1)
        params[3] = np.tanh(params[3])

        return params

    def untransform_params(self, constrained):

        params = constrained.copy()

        params[:3] = np.sqrt(params[:3])

        params[3] = np.arctanh(params[3])

        return params

    def update(self, params, *args, **kwargs):

        params = super().update(params, *args, **kwargs)

        sigma2_level = params[0]
        sigma2_trend = params[1]
        sigma2_error = params[2]
        phi = params[3]

        # AR(1) coefficient
        self.ssm["transition", 2, 2] = phi

        # State covariance
        self.ssm["state_cov", 0, 0] = sigma2_level
        self.ssm["state_cov", 1, 1] = sigma2_trend
        self.ssm["state_cov", 2, 2] = sigma2_error

        return params