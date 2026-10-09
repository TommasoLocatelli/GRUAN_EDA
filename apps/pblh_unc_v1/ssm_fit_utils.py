
import io
import warnings
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde, norm, probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


def report_smoothing_goodness(results, observations, path):
    """
    Create a PDF diagnostic report for fitted state-space model results.

    Parameters
    ----------
    results : statsmodels state-space results
        Fitted results, preferably with smoothing enabled.
    observations : array-like
        Observed time series, shape (nobs,) or (nobs, k_endog).
    path : str or path-like
        Output PDF filename.

    Report sections
    ---------------
    1. Convergence and model information
    2. Parameter estimates and uncertainty
    3. Observations with measurement uncertainty
    4. Smoothed states with uncertainty
    5. Smoothed measurement disturbances with uncertainty
    6. Standardized residual histogram and normal comparison
    7. Normal Q-Q plot
    8. Residual autocorrelation
    9. Residual metrics and statistical tests

    Notes
    -----
    State-space arrays commonly use the shape
    (number of variables, number of observations).
    The function handles this convention explicitly.

    Forecast errors are not the same as smoothed disturbances.
    Both are labeled according to their actual statistical meaning.
    """

    y = np.asarray(observations, dtype=float)

    if y.ndim == 1:
        y = y[:, None]

    if y.ndim != 2:
        raise ValueError(
            "observations must be a 1D or 2D array."
        )

    nobs, k_endog = y.shape

    if nobs == 0:
        raise ValueError("observations cannot be empty.")

    model = results.model
    fr = results.filter_results

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def get_array(name):
        """Return an available result array, or None."""
        value = getattr(results, name, None)
        if value is None:
            value = getattr(fr, name, None)
        if value is None:
            return None
        return np.asarray(value)

    def observation_by_time(arr, n, k):
        """
        Convert an observation-space array to shape (n, k).
        Expected input shape is generally (k, n).
        """
        if arr is None:
            return None

        arr = np.asarray(arr)

        if arr.ndim == 1:
            if arr.size == n:
                return arr[:, None]
            return None

        if arr.ndim == 2:
            if arr.shape == (k, n):
                return arr.T
            if arr.shape == (n, k):
                return arr

        return None

    def safe_float(value):
        try:
            value = np.asarray(value).squeeze()
            return float(value) if value.ndim == 0 else str(value)
        except Exception:
            return str(value)

    def add_text_page(title, lines):
        """Write readable text onto one or more PDF pages."""
        lines_per_page = 48

        for start in range(0, max(len(lines), 1), lines_per_page):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle(
                title if start == 0 else title + " (continued)",
                fontsize=15,
                y=0.97
            )

            page_lines = lines[start:start + lines_per_page]

            fig.text(
                0.07, 0.92,
                "\n".join(page_lines),
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                wrap=True
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    def add_series_plot(
        title, series, labels, ylabel,
        lower=None, upper=None
    ):
        """Plot one or more time series with optional uncertainty bands."""
        series = np.asarray(series, dtype=float)

        if series.ndim == 1:
            series = series[:, None]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        t = np.arange(series.shape[0])

        for j in range(series.shape[1]):
            label = (
                labels[j]
                if j < len(labels)
                else f"Series {j + 1}"
            )

            ax.plot(t, series[:, j], label=label, linewidth=1.2)

            if lower is not None and upper is not None:
                lo = np.asarray(lower)
                hi = np.asarray(upper)

                if lo.ndim == 1:
                    lo = lo[:, None]
                if hi.ndim == 1:
                    hi = hi[:, None]

                if (
                    j < lo.shape[1]
                    and j < hi.shape[1]
                    and lo.shape[0] == series.shape[0]
                    and hi.shape[0] == series.shape[0]
                ):
                    ax.fill_between(
                        t, lo[:, j], hi[:, j],
                        alpha=0.2
                    )

        ax.set_title(title)
        ax.set_xlabel("Observation")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def covariance_std(cov, n, k):
        """
        Extract marginal standard deviations from covariance arrays.

        Supports common observation-space layouts:
        (k, k, n), (n, k, k), or (k, n).
        """
        if cov is None:
            return None

        cov = np.asarray(cov)

        if cov.ndim == 3:
            if cov.shape == (k, k, n):
                variances = np.stack(
                    [np.diag(cov[:, :, i]) for i in range(n)]
                )
            elif cov.shape == (n, k, k):
                variances = np.stack(
                    [np.diag(cov[i]) for i in range(n)]
                )
            else:
                return None

        elif cov.ndim == 2 and cov.shape == (k, n):
            variances = cov.T

        else:
            return None

        return np.sqrt(np.maximum(variances, 0))

    def state_std(cov):
        """Extract standard deviations for states over time."""
        if cov is None:
            return None

        cov = np.asarray(cov)

        if cov.ndim != 3:
            return None

        # statsmodels state covariance convention: (states, states, time)
        if cov.shape[0] == cov.shape[1]:
            return np.sqrt(
                np.maximum(
                    np.diagonal(cov, axis1=0, axis2=1).T,
                    0
                )
            )

        # Alternative layout: (time, states, states)
        if cov.shape[1] == cov.shape[2]:
            return np.sqrt(
                np.maximum(
                    np.diagonal(cov, axis1=1, axis2=2),
                    0
                )
            )

        return None

    # ---------------------------------------------------------
    # Gather model and smoother output
    # ---------------------------------------------------------

    smoothed_state = get_array("smoothed_state")
    smoothed_state_cov = get_array("smoothed_state_cov")

    smoothed_measurement = get_array(
        "smoothed_measurement_disturbance"
    )
    smoothed_measurement_cov = get_array(
        "smoothed_measurement_disturbance_cov"
    )

    standardized_errors = get_array(
        "standardized_forecasts_error"
    )
    forecast_errors = get_array("forecasts_error")
    forecast_error_cov = get_array("forecasts_error_cov")

    obs_cov = getattr(model.ssm, "obs_cov", None)
    obs_cov = None if obs_cov is None else np.asarray(obs_cov)

    state_names = list(
        getattr(model, "state_names", []) or []
    )
    if smoothed_state is not None:
        n_states = smoothed_state.shape[0]
        if len(state_names) != n_states:
            state_names = [
                f"State {i + 1}" for i in range(n_states)
            ]

    endog_names = getattr(model, "endog_names", None)
    if not isinstance(endog_names, list):
        endog_names = [endog_names] if endog_names else []

    if len(endog_names) != k_endog:
        endog_names = [
            f"Observation {i + 1}" for i in range(k_endog)
        ]

    # ---------------------------------------------------------
    # Build the report
    # ---------------------------------------------------------

    with PdfPages(path) as pdf:

        # 1. Convergence and model summary
        lines = [
            "MODEL",
            f"Model class: {model.__class__.__name__}",
            f"Observations supplied: {nobs}",
            f"Endogenous variables: {k_endog}",
            f"Endogenous names: {endog_names}",
            f"State names: {state_names}",
            f"Log likelihood: {safe_float(getattr(results, 'llf', np.nan))}",
            f"AIC: {safe_float(getattr(results, 'aic', np.nan))}",
            f"BIC: {safe_float(getattr(results, 'bic', np.nan))}",
            f"HQIC: {safe_float(getattr(results, 'hqic', np.nan))}",
            "",
            "CONVERGENCE",
        ]

        mle_retvals = getattr(results, "mle_retvals", None)

        if mle_retvals is not None:
            for key, value in mle_retvals.items():
                lines.append(f"{key}: {value}")
        else:
            lines.append(
                "Optimizer return details are unavailable."
            )

        mle_settings = getattr(results, "mle_settings", None)
        if mle_settings:
            lines.append("")
            lines.append("Optimizer settings:")
            for key, value in mle_settings.items():
                lines.append(f"{key}: {value}")

        lines.extend([
            "",
            "MODEL SPECIFICATION",
            f"Number of estimated parameters: {len(results.params)}",
            f"Parameter names: {getattr(results, 'param_names', [])}",
            f"Number of diffuse observations: "
            f"{getattr(fr, 'nobs_diffuse', 'unavailable')}",
            f"Likelihood burn: "
            f"{getattr(results, 'loglikelihood_burn', 'unavailable')}",
        ])

        add_text_page("State-Space Smoothing Report", lines)

        # 2. Parameter estimates and confidence intervals
        try:
            param_table = results.summary().tables
            for table in param_table:
                add_text_page(
                    "Parameter Estimates",
                    str(table).splitlines()
                )
        except Exception as exc:
            lines = ["Parameter estimates"]
            names = getattr(
                results, "param_names",
                [f"Parameter {i + 1}" for i in range(len(results.params))]
            )

            try:
                bse = np.asarray(results.bse)
            except Exception:
                bse = np.full(len(results.params), np.nan)

            for i, value in enumerate(results.params):
                se = bse[i] if i < len(bse) else np.nan
                lines.append(
                    f"{names[i] if i < len(names) else i}: "
                    f"estimate={value:.6g}, SE={se:.6g}"
                )

            lines.append(f"Summary fallback reason: {exc}")
            add_text_page("Parameter Estimates", lines)

        # 3. Observations with measurement uncertainty
        # Measurement covariance describes observation noise, not
        # uncertainty in the observed data themselves.
        obs_sd = None

        if obs_cov is not None:
            if obs_cov.ndim == 3:
                if obs_cov.shape[:2] == (k_endog, k_endog):
                    obs_sd = np.sqrt(
                        np.maximum(
                            np.stack([
                                np.diag(obs_cov[:, :, i])
                                for i in range(obs_cov.shape[2])
                            ]),
                            0
                        )
                    )
            elif obs_cov.ndim == 2 and obs_cov.shape == (
                k_endog, k_endog
            ):
                obs_sd = np.tile(
                    np.sqrt(np.maximum(np.diag(obs_cov), 0)),
                    (nobs, 1)
                )

        if obs_sd is not None:
            obs_sd = obs_sd[:nobs]
            if obs_sd.shape[0] == nobs:
                add_series_plot(
                    "Observed Series with Measurement-Noise Band (±1 SD)",
                    y,
                    endog_names,
                    "Observed value",
                    y - obs_sd,
                    y + obs_sd
                )
        else:
            add_series_plot(
                "Observed Series",
                y,
                endog_names,
                "Observed value"
            )

        # 4. Smoothed states with uncertainty
        if smoothed_state is not None:
            states = np.asarray(smoothed_state).T

            std = state_std(smoothed_state_cov)
            if std is not None:
                std = std[:states.shape[0]]

            if std is not None and std.shape == states.shape:
                add_series_plot(
                    "Smoothed State Estimates with 95% Confidence Bands",
                    states,
                    state_names,
                    "Smoothed state",
                    states - 1.96 * std,
                    states + 1.96 * std
                )
            else:
                add_series_plot(
                    "Smoothed State Estimates (Uncertainty Unavailable)",
                    states,
                    state_names,
                    "Smoothed state"
                )
        else:
            add_text_page(
                "Smoothed States",
                ["Smoothed states are unavailable. Ensure the model was smoothed."]
            )

        # 5. Smoothed measurement disturbances
        if smoothed_measurement is not None:
            disturbance = observation_by_time(
                smoothed_measurement, nobs, k_endog
            )

            disturbance_std = covariance_std(
                smoothed_measurement_cov, nobs, k_endog
            )

            if disturbance is not None:
                if (
                    disturbance_std is not None
                    and disturbance_std.shape == disturbance.shape
                ):
                    add_series_plot(
                        "Smoothed Measurement Disturbances with 95% Bands",
                        disturbance,
                        endog_names,
                        "Smoothed measurement disturbance",
                        disturbance - 1.96 * disturbance_std,
                        disturbance + 1.96 * disturbance_std
                    )
                else:
                    add_series_plot(
                        "Smoothed Measurement Disturbances",
                        disturbance,
                        endog_names,
                        "Smoothed measurement disturbance"
                    )
        else:
            add_text_page(
                "Smoothed Measurement Disturbances",
                [
                    "Smoothed measurement disturbances are unavailable.",
                    "These are not automatically equivalent to residuals",
                    "from the smoothed state signal."
                ]
            )

        # 6. Standardized residual diagnostics
        # These are standardized one-step-ahead forecast errors.
        # statsmodels does not generally provide a universal
        # standardized smoothed-residual series.
        std_err = observation_by_time(
            standardized_errors, nobs, k_endog
        )

        if std_err is None:
            add_text_page(
                "Residual Diagnostics",
                [
                    "Standardized forecast errors are unavailable.",
                    "Histogram, Q-Q plot, and autocorrelation diagnostics",
                    "cannot be computed from this results object."
                ]
            )
            return

        for j in range(k_endog):
            r = std_err[:, j]
            r = r[np.isfinite(r)]

            if r.size < 3:
                continue

            name = endog_names[j]

            # Exclude diffuse initialization and likelihood-burn periods
            burn = max(
                int(getattr(results, "loglikelihood_burn", 0) or 0),
                int(getattr(fr, "nobs_diffuse", 0) or 0)
            )
            r_all = std_err[:, j]
            r = r_all[burn:]
            r = r[np.isfinite(r)]

            if r.size < 3:
                continue

            # Histogram and normal density
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(
                r, bins="auto", density=True,
                alpha=0.55, edgecolor="black",
                label="Standardized forecast errors"
            )

            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 300)
            ax.plot(
                x, norm.pdf(x), linewidth=2,
                label="Standard normal N(0, 1)"
            )

            if r.size >= 5 and np.std(r) > 0:
                try:
                    kde = gaussian_kde(r)
                    ax.plot(x, kde(x), label="Estimated density")
                except (ValueError, np.linalg.LinAlgError):
                    pass

            ax.set_title(f"Standardized Residual Distribution: {name}")
            ax.set_xlabel("Standardized forecast error")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Q-Q plot
            fig, ax = plt.subplots(figsize=(6, 6))
            probplot(r, dist="norm", plot=ax)
            ax.set_title(f"Normal Q-Q Plot: {name}")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ACF plot with approximate confidence bands
            max_lag = min(40, max(1, len(r) // 5))
            centered = r - np.mean(r)
            denom = np.dot(centered, centered)

            if denom > 0:
                acf = np.array([
                    np.dot(centered[lag:], centered[:len(r)-lag])
                    / denom
                    for lag in range(max_lag + 1)
                ])

                fig, ax = plt.subplots(figsize=(9, 4.5))
                lags = np.arange(max_lag + 1)
                ax.axhline(0, linewidth=1)
                ax.axhline(
                    1.96 / np.sqrt(len(r)),
                    linestyle="--"
                )
                ax.axhline(
                    -1.96 / np.sqrt(len(r)),
                    linestyle="--"
                )
                ax.vlines(lags[1:], 0, acf[1:])
                ax.plot(lags, acf, "o", markersize=3)
                ax.set_title(
                    f"Standardized Forecast-Error Autocorrelation: {name}"
                )
                ax.set_xlabel("Lag")
                ax.set_ylabel("Autocorrelation")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # 7. Residual metrics and statistical tests
            lines = [
                f"Residual diagnostics: {name}",
                f"Number of usable residuals: {len(r)}",
                f"Mean: {np.mean(r):.6g}",
                f"Standard deviation: {np.std(r, ddof=1):.6g}",
                f"Mean absolute error: {np.mean(np.abs(r)):.6g}",
                f"Mean squared error: {np.mean(r ** 2):.6g}",
                f"Root mean squared error: "
                f"{np.sqrt(np.mean(r ** 2)):.6g}",
                f"Minimum: {np.min(r):.6g}",
                f"Maximum: {np.max(r):.6g}",
                f"Skewness/kurtosis/JB test:",
            ]

            try:
                jb, jb_pvalue, skew, kurtosis = jarque_bera(r)
                lines.extend([
                    f"Jarque-Bera statistic: {jb:.6g}",
                    f"Jarque-Bera p-value: {jb_pvalue:.6g}",
                    f"Skewness: {skew:.6g}",
                    f"Kurtosis: {kurtosis:.6g}",
                ])
            except Exception as exc:
                lines.append(f"Jarque-Bera unavailable: {exc}")

            if len(r) >= 10:
                try:
                    lb_lags = sorted(set([
                        min(5, len(r) // 5),
                        min(10, len(r) // 5)
                    ]))
                    lb_lags = [lag for lag in lb_lags if lag >= 1]

                    lb = acorr_ljungbox(
                        r, lags=lb_lags, return_df=True
                    )

                    lines.append("")
                    lines.append("Ljung-Box serial-correlation test:")

                    for lag, row in lb.iterrows():
                        lines.append(
                            f"Lag {lag}: Q={row['lb_stat']:.6g}, "
                            f"p-value={row['lb_pvalue']:.6g}"
                        )
                except Exception as exc:
                    lines.append(
                        f"Ljung-Box test unavailable: {exc}"
                    )

            add_text_page("Residual Metrics", lines)

    print(f"Saved report: {path}")
