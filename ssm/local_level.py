import numpy as np
import statsmodels.api as sm

class MeasurementLocalLevel(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, measurement_var, measurement_names):
        # Model order
        k_states = endog.shape[1] # The dimension of the unobserved state process
        k_posdef = endog.shape[1] # The dimension of a guaranteed positive definite covariance matrix describing the shocks in the transition equation.

        # Store fixed measurement variance sequence if provided
        self.measurement_var = measurement_var

        # Store measurement names for later use
        self.measurement_names = measurement_names

        # Initialize the statespace
        super(MeasurementLocalLevel, self).__init__(
            endog, # The observed time-series process
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # Initialize the matrices
        # The design matrix maps the state space to the observed data
        self.ssm["design"] = np.eye(k_states)
        # The transition matrix defines how the state evolves over time
        self.ssm["transition"] = np.eye(k_states)
        # The selection matrix maps the shocks to the state space
        self.ssm["selection"] = np.eye(k_states)

        # Use time-varying observation covariance 
        self.ssm["obs_cov"] = np.zeros((k_states, k_states, self.nobs))

        # Cache some indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return [f"{name}_sigma2.level" for name in self.measurement_names ]

    @property
    def start_params(self):
        return [np.std(self.endog)] * len(self.measurement_names)

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(MeasurementLocalLevel, self).update(params, *args, **kwargs)

        for i, name in enumerate(self.measurement_names):
            self.ssm["obs_cov", i, i, :] = self.measurement_var[:, i]
        state_params = params

        # State covariance
        self.ssm[self._state_cov_idx] = state_params

