import numpy as np
import statsmodels.api as sm
class RHLocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, alt_measurement_var, rh_measurement_var):
        # Model order
        k_states = 4 # The dimension of the unobserved state process
        k_posdef = 4 # The dimension of a guaranteed positive definite covariance matrix describing the shocks in the transition equation.

        # Store fixed measurement variance sequence if provided
        self.alt_measurement_var = np.asarray(alt_measurement_var)
        self.rh_measurement_var = np.asarray(rh_measurement_var)

        # Initialize the statespace
        super(RHLocalLinearTrend, self).__init__(
            endog, # The observed time-series process
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # Initialize the matrices
        self.ssm["design"] = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) # The design matrix maps the state space to the observed data
        self.ssm["transition"] = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]) # The transition matrix defines how the state evolves over time
        self.ssm["selection"] = np.eye(k_states) # The selection matrix maps the shocks to the state space

        # Use time-varying observation covariance 
        self.ssm["obs_cov"] = np.zeros((2, 2, self.nobs))

        # Cache some indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["alt_sigma2.level", "alt_sigma2.trend", "rh_sigma2.level", "rh_sigma2.trend"]

    @property
    def start_params(self):
        return [np.std(self.endog)] * 4

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(RHLocalLinearTrend, self).update(params, *args, **kwargs)

        self.ssm["obs_cov", 0, 0, :] = self.alt_measurement_var
        self.ssm["obs_cov", 1, 1, :] = self.rh_measurement_var # check
        state_params = params

        # State covariance
        self.ssm[self._state_cov_idx] = state_params