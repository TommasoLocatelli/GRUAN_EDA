import numpy as np
import statsmodels.api as sm

# State vector
# [z, Lz, Th, LTh, RH, LRH, u, Lu, v, Lv]
# Observation vector
# [z, Th, RH, u, v]

# Transition matrix
PHI=np.array([  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
# Observation matrix
A=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

class PreTransformedLocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, measurement_var):
        # Model order
        k_states = 10 # The dimension of the unobserved state process
        k_posdef = 10 # The dimension of a guaranteed positive definite covariance matrix describing the shocks in the transition equation.

        # Store fixed measurement variance sequence if provided
        self.measurement_var = np.asarray(measurement_var)

        # Initialize the statespace
        super(PreTransformedLocalLinearTrend, self).__init__(
            endog, # The observed time-series process
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        # Initialize the matrices
        self.ssm["design"] = A # The design matrix maps the state space to the observed data
        self.ssm["transition"] = PHI # The transition matrix defines how the state evolves over time
        self.ssm["selection"] = np.eye(k_states) # The selection matrix maps the shocks to the state space

        # Use time-varying observation covariance 
        self.ssm["obs_cov"] = np.zeros((5, 5, self.nobs))

        # Cache some indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["alt_sigma2.level", "alt_sigma2.trend",
                "th_sigma2.level", "th_sigma2.trend", 
                "rh_sigma2.level", "rh_sigma2.trend",
                "u_sigma2.level", "u_sigma2.trend", 
                "v_sigma2.level", "v_sigma2.trend"]

    @property
    def start_params(self):
        return [np.std(self.endog)] * 10

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5


    def update(self, params, *args, **kwargs):
        # Call parent update (handles transformed params)
        params = super(PreTransformedLocalLinearTrend, self).update(
            params, *args, **kwargs
        )

        # ----------------------------------------------------
        # 1. Update observation covariance (time-varying)
        # ----------------------------------------------------
        # measurement_var is a vector of length 5: [z, Th, RH, u, v]
        # A is 5×10, so obs_cov must be 5×5×T
        for i in range(5):
            self.ssm["obs_cov", i, i, :] = self.measurement_var[i]

        # ----------------------------------------------------
        # 2. Update state covariance (10×10 diagonal)
        # ----------------------------------------------------
        # params already transformed (σ²)
        state_params = params
        self.ssm[self._state_cov_idx] = state_params
