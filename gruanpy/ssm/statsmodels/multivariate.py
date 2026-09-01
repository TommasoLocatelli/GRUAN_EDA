import statsmodels.api as sm
import numpy as np

class MultivariateLLT(sm.tsa.statespace.MLEModel):
    """
    Multivariate Local Linear Trend Model (MLLT)
    y_t = level_t + eps_t
    level_t = level_{t-1} + slope_{t-1} + eta_t
    slope_t = slope_{t-1} + zeta_t
    """
    def __init__(self, endog):
        k_endog = endog.shape[1]
        k_states = 2 * k_endog
        k_posdef = 2 * k_endog

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # Design matrix: each variable observes its own level
        design = np.zeros((k_endog, k_states))
        for i in range(k_endog):
            design[i, 2*i] = 1.0
        self.ssm["design"] = design

        # Transition matrix: block-diagonal LLT
        transition = np.zeros((k_states, k_states))
        for i in range(k_endog):
            transition[2*i, 2*i] = 1.0
            transition[2*i, 2*i+1] = 1.0
            transition[2*i+1, 2*i+1] = 1.0
        self.ssm["transition"] = transition

        # Selection matrix: identity
        self.ssm["selection"] = np.eye(k_states)

        # Cache diagonal indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["sigma2.measurement_" + str(i) for i in range(self.k_endog)] + \
               ["sigma2.level_" + str(i) for i in range(self.k_endog)] + \
               ["sigma2.trend_" + str(i) for i in range(self.k_endog)]

    @property
    def start_params(self):
        return [np.std(self.endog[:, i]) for i in range(self.k_endog)] * 3

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)

        k = self.k_endog

        # Measurement noise
        for i in range(k):
            self.ssm["obs_cov", i, i] = params[i]

        # State noise (level + trend)
        self.ssm[self._state_cov_idx] = params[k:]

class MultivariateLLL(sm.tsa.statespace.MLEModel):
    """
    Multivariate Local Linear Level Model (MLLL)
    y_t = level_t + eps_t
    level_t = level_{t-1} + eta_t
    """
    def __init__(self, endog):
        k_endog = endog.shape[1]
        k_states = k_endog
        k_posdef = k_endog

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # Design: each variable observes its own level
        self.ssm["design"] = np.eye(k_endog)

        # Transition: random walk for each level
        self.ssm["transition"] = np.eye(k_states)

        # Selection: identity
        self.ssm["selection"] = np.eye(k_states)

        # Cache diagonal indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["sigma2.measurement_" + str(i) for i in range(self.k_endog)] + \
               ["sigma2.level_" + str(i) for i in range(self.k_endog)]

    @property
    def start_params(self):
        return [np.std(self.endog[:, i]) for i in range(self.k_endog)] * 2

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)

        k = self.k_endog

        # Measurement noise
        for i in range(k):
            self.ssm["obs_cov", i, i] = params[i]

        # State noise (level innovations)
        self.ssm[self._state_cov_idx] = params[k:]
