import statsmodels.api as sm
import numpy as np

def prepare_measurement_sigma2(measurement_sigma2, nobs, k_endog):
    if measurement_sigma2 is None:
        return None

    variance = np.asarray(measurement_sigma2, dtype=float)

    if variance.shape == (nobs, k_endog):
        variance = variance.T
    elif variance.shape != (k_endog, nobs):
        raise ValueError(
            "measurement_sigma2 must have shape "
            f"({nobs}, {k_endog}) or ({k_endog}, {nobs}); "
            f"got {variance.shape}"
        )

    if not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError(
            "measurement_sigma2 must contain finite, nonnegative variances"
        )

    return variance.copy()


class MultivariateLLT(sm.tsa.statespace.MLEModel):
    """
    Multivariate Local Linear Trend Model.

    y_t     = level_t + eps_t
    level_t = level_{t-1} + slope_{t-1} + eta_t
    slope_t = slope_{t-1} + zeta_t

    measurement_sigma2:
        Optional known measurement variances, shape (nobs, k_endog)
        or (k_endog, nobs).
    """

    def __init__(self, endog, measurement_sigma2=None):
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

        self.measurement_sigma2 = prepare_measurement_sigma2(
            measurement_sigma2, self.nobs, k_endog
        )

        design = np.zeros((k_endog, k_states))
        for i in range(k_endog):
            design[i, 2 * i] = 1.0
        self.ssm["design"] = design

        transition = np.zeros((k_states, k_states))
        for i in range(k_endog):
            transition[2 * i, 2 * i] = 1.0
            transition[2 * i, 2 * i + 1] = 1.0
            transition[2 * i + 1, 2 * i + 1] = 1.0
        self.ssm["transition"] = transition

        self.ssm["selection"] = np.eye(k_states)

        # Explicitly allocate time-varying observation covariance.
        self.ssm["obs_cov"] = np.zeros(
            (k_endog, k_endog, self.nobs)
        )

        self._state_cov_idx = (
            "state_cov", np.diag_indices(k_posdef)
        )

    @property
    def param_names(self):
        k = self.k_endog

        measurement_names = (
            [f"sigma2.measurement_{i}" for i in range(k)]
            if self.measurement_sigma2 is None else []
        )

        return measurement_names + (
            [f"sigma2.level_{i}" for i in range(k)]
            + [f"sigma2.trend_{i}" for i in range(k)]
        )

    @property
    def start_params(self):
        k = self.k_endog
        scales = [
            max(float(np.nanstd(self.endog[:, i])), 1e-3)
            for i in range(k)
        ]

        if self.measurement_sigma2 is None:
            return scales + scales + scales

        return scales + scales

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)
        k = self.k_endog

        if self.measurement_sigma2 is None:
            measurement_params = params[:k]
            state_params = params[k:]

            for i in range(k):
                self.ssm["obs_cov", i, i] = measurement_params[i]
        else:
            state_params = params

            for i in range(k):
                self.ssm["obs_cov", i, i] = self.measurement_sigma2[i, :]

        level_params = state_params[:k]
        trend_params = state_params[k:]

        for i in range(k):
            self.ssm["state_cov", 2*i, 2*i] = level_params[i]
            self.ssm["state_cov", 2*i+1, 2*i+1] = trend_params[i]
class MultivariateLLL(sm.tsa.statespace.MLEModel):
    """
    Multivariate Local Level Model.

    y_t     = level_t + eps_t
    level_t = level_{t-1} + eta_t
    """

    def __init__(self, endog, measurement_sigma2=None):
        k_endog = endog.shape[1]

        super().__init__(
            endog,
            k_states=k_endog,
            k_posdef=k_endog,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_endog,
        )

        self.measurement_sigma2 = prepare_measurement_sigma2(
            measurement_sigma2, self.nobs, k_endog
        )

        self.ssm["design"] = np.eye(k_endog)
        self.ssm["transition"] = np.eye(k_endog)
        self.ssm["selection"] = np.eye(k_endog)

        self.ssm["obs_cov"] = np.zeros(
            (k_endog, k_endog, self.nobs)
        )

        self._state_cov_idx = (
            "state_cov", np.diag_indices(k_endog)
        )

    @property
    def param_names(self):
        k = self.k_endog

        measurement_names = (
            [f"sigma2.measurement_{i}" for i in range(k)]
            if self.measurement_sigma2 is None else []
        )

        return measurement_names + [
            f"sigma2.level_{i}" for i in range(k)
        ]

    @property
    def start_params(self):
        scales = [
            max(float(np.nanstd(self.endog[:, i])), 1e-3)
            for i in range(self.k_endog)
        ]

        if self.measurement_sigma2 is None:
            return scales + scales

        return scales

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)
        k = self.k_endog

        if self.measurement_sigma2 is None:
            measurement_params = params[:k]
            state_params = params[k:]

            for i in range(k):
                self.ssm["obs_cov", i, i] = measurement_params[i]
        else:
            state_params = params

            for i in range(k):
                self.ssm["obs_cov", i, i] = self.measurement_sigma2[i, :]

        # Explicitly assign each level variance.
        for i in range(k):
            self.ssm["state_cov", i, i] = state_params[i]

class MultivariateLLTDetLevel(sm.tsa.statespace.MLEModel):
    """
    Multivariate Local Linear Trend with deterministic level.

    y_t     = level_t + eps_t
    level_t = level_{t-1} + slope_{t-1}
    slope_t = slope_{t-1} + zeta_t
    """

    def __init__(self, endog, measurement_sigma2=None):
        k_endog = endog.shape[1]
        k_states = 2 * k_endog
        k_posdef = k_endog

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=k_states,
        )

        self.measurement_sigma2 = prepare_measurement_sigma2(
            measurement_sigma2, self.nobs, k_endog
        )

        design = np.zeros((k_endog, k_states))
        for i in range(k_endog):
            design[i, 2 * i] = 1.0
        self.ssm["design"] = design

        transition = np.zeros((k_states, k_states))
        for i in range(k_endog):
            transition[2 * i, 2 * i] = 1.0
            transition[2 * i, 2 * i + 1] = 1.0
            transition[2 * i + 1, 2 * i + 1] = 1.0
        self.ssm["transition"] = transition

        selection = np.zeros((k_states, k_posdef))
        for i in range(k_endog):
            selection[2 * i + 1, i] = 1.0
        self.ssm["selection"] = selection

        self.ssm["obs_cov"] = np.zeros(
            (k_endog, k_endog, self.nobs)
        )

        self._state_cov_idx = (
            "state_cov", np.diag_indices(k_posdef)
        )

    @property
    def param_names(self):
        k = self.k_endog

        measurement_names = (
            [f"sigma2.measurement_{i}" for i in range(k)]
            if self.measurement_sigma2 is None else []
        )

        return measurement_names + [
            f"sigma2.trend_{i}" for i in range(k)
        ]

    @property
    def start_params(self):
        scales = [
            max(float(np.nanstd(self.endog[:, i])), 1e-3)
            for i in range(self.k_endog)
        ]

        if self.measurement_sigma2 is None:
            return scales + [0.1 * s for s in scales]

        return [0.1 * s for s in scales]

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)
        k = self.k_endog

        if self.measurement_sigma2 is None:
            measurement_params = params[:k]
            trend_params = params[k:]

            for i in range(k):
                self.ssm["obs_cov", i, i] = measurement_params[i]
        else:
            trend_params = params

            for i in range(k):
                self.ssm["obs_cov", i, i] = self.measurement_sigma2[i, :]

        for i in range(k):
            self.ssm["state_cov", i, i] = trend_params[i]
        """
        Multivariate Local Linear Trend with deterministic level

        y_t = level_t + eps_t

        level_t = level_{t-1} + slope_{t-1}

        slope_t = slope_{t-1} + zeta_t
        """

        def __init__(self, endog):
            k_endog = endog.shape[1]

            k_states = 2 * k_endog
            k_posdef = k_endog  # only trend shocks

            super().__init__(
                endog,
                k_states=k_states,
                k_posdef=k_posdef,
                initialization="approximate_diffuse",
                loglikelihood_burn=k_states,
            )

            # Design matrix
            design = np.zeros((k_endog, k_states))
            for i in range(k_endog):
                design[i, 2 * i] = 1.0
            self.ssm["design"] = design

            # Transition matrix
            transition = np.zeros((k_states, k_states))
            for i in range(k_endog):
                transition[2*i, 2*i] = 1.0
                transition[2*i, 2*i+1] = 1.0
                transition[2*i+1, 2*i+1] = 1.0

            self.ssm["transition"] = transition

            # Selection matrix:
            # disturbances affect only slopes
            selection = np.zeros((k_states, k_endog))
            for i in range(k_endog):
                selection[2*i+1, i] = 1.0

            self.ssm["selection"] = selection

            self._state_cov_idx = ("state_cov",) + np.diag_indices(k_endog)

        @property
        def param_names(self):
            return (
                [f"sigma2.measurement_{i}" for i in range(self.k_endog)]
                + [f"sigma2.trend_{i}" for i in range(self.k_endog)]
            )

        @property
        def start_params(self):
            return (
                [np.nanstd(self.endog[:, i]) for i in range(self.k_endog)]
                + [0.1 * np.nanstd(self.endog[:, i]) for i in range(self.k_endog)]
            )

        def transform_params(self, unconstrained):
            return unconstrained**2

        def untransform_params(self, constrained):
            return constrained**0.5

        def update(self, params, *args, **kwargs):
            params = super().update(params, *args, **kwargs)

            k = self.k_endog

            # Measurement variances
            for i in range(k):
                self.ssm["obs_cov", i, i] = params[i]

            # Trend disturbance variances
            self.ssm[self._state_cov_idx] = params[k:]