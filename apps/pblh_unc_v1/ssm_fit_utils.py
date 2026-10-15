
import io
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde, norm, probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import jarque_bera
import pprint

def add_text_page(pdf, text, title=None, fontsize=8):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    ax.axis("off")

    y = 0.98

    if title:
        ax.text(
            0.01, y, title,
            fontsize=12,
            fontweight="bold",
            va="top",
            family="monospace"
        )
        y -= 0.04

    ax.text(
        0.01, y, text,
        fontsize=fontsize,
        va="top",
        family="monospace",
        wrap=True
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def report_smoothing_goodness(
        results,
        observations,
        meas_unc=None,
        path='test.pdf',
        plot=True,
        zoom=False):
    """
    Generate a PDF report assessing the quality of a state-space model fit
    and Kalman smoothing results.

    The report summarizes the fitted model, optimization diagnostics,
    smoothed state estimates, residual behaviour, uncertainty calibration,
    and goodness-of-fit metrics. It is intended as a post-fit diagnostic
    tool to evaluate whether the smoothed state provides an adequate
    representation of the observed data.

    The report includes:

    - Model summary produced by Statsmodels.
    - MLE optimization settings and convergence information.
    - Comparison between observations and smoothed state estimates.
    - Smoothed-state uncertainty intervals.
    - Residual and standardized residual diagnostics.
    - Goodness-of-fit metrics (RMSE, MAE, Bias, R²).
    - Standardized residual coverage diagnostics.
    - Information criteria (Log-Likelihood, AIC, BIC).
    - Residual independence tests (Ljung-Box).
    - Residual normality test (Jarque-Bera).

    Notes
    -----
    Residuals analyzed in this report are smoothing residuals:

        residual = observation - smoothed_state

    These residuals assess how closely the smoothed state reproduces
    the observed data. They differ from the one-step-ahead prediction
    errors (innovations) typically reported in the Statsmodels model
    summary.

    Parameters
    ----------
    results : statsmodels.tsa.statespace.mlemodel.MLEResults
        Fitted state-space model results object.

    observations : array-like
        Original observations used to fit the model.

    meas_unc : float or array-like, optional
        Observation uncertainty (1σ). If the model estimates
        'sigma2.measurement', the fitted value is used instead.

    path : str, default='test.pdf'
        Output PDF filename.

    plot : bool, default=True
        If True, figures are displayed interactively before being
        written to the PDF report.

    zoom : bool or tuple(int, int), default=False
        If a tuple (xmin, xmax) is provided, the observation versus
        smoothed-state plot is restricted to the specified interval.

    Returns
    -------
    None
        Saves the generated report to 'path'.

    Examples
    --------
    >>> results = model.fit()
    >>> report_smoothing_goodness(
    ...     results,
    ...     observations=y,
    ...     meas_unc=0.5,
    ...     path="smoothing_report.pdf"
    ... )
    """

    # ---------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------
    obs = np.asarray(observations, dtype=float)
    if meas_unc is not None:
        meas_unc = np.asarray(meas_unc, dtype=float)
    model = results.model
    
    nobs = results.nobs
    params = results.params
    if "sigma2.measurement" in params:
        prior_meas_unc_flag = False
        sigma2_measurement=params['sigma2.measurement']
        meas_unc=(sigma2_measurement**0.5)*2
    else:
        prior_meas_unc_flag = True

    smoothed_state=results.smoothed_state[0]
    smoothed_state_cov=results.smoothed_state_cov
    smoothed_var = smoothed_state_cov[0, 0, :]
    smoothed_std = np.sqrt(smoothed_var)

    # residuals
    smoothed_resid = obs - smoothed_state

    # standardized residuals
    smoothed_std_resid = smoothed_resid / np.maximum(smoothed_std, 1e-8)

    ret = results.mle_retvals
    sett = results.mle_settings
    method = sett.get("optimizer", "unknown")
    converged = ret.get("converged", "unknown")
    iterations = ret.get("iterations", "unknown")
    fcalls = ret.get("fcalls", "unknown")
    warnflag = ret.get("warnflag", "unknown")
    fopt = ret.get("fopt", np.nan)

    mle_text = f"""
    Optimizer
    ---------
    Method               : {method}

    Convergence
    -----------
    Converged            : {converged}
    Iterations           : {iterations}
    Function evaluations : {fcalls}
    Warning flag         : {warnflag}

    Objective Function
    ------------------
    Final objective      : {fopt:.6f}

    Interpretation
    --------------
    Converged=True and Warning flag=0 indicate that the optimizer
    terminated successfully and a local optimum was found.
    """


    summary=results.summary()

    summary_text = results.summary().as_text()
    summary_text += "\n\nNote:\n- State Space Model Summary Residuals in diagnostic tests are one-step-ahead prediction errors. Futher analysis will consider smoothed states residuals."
    if prior_meas_unc_flag:
        summary_text += "\n- Measurement unceratinty is fixed before MLE."
    else:
        summary_text += "\n- Measurement unceratinty is estimated by MLE."

    aic = results.aic
    bic = results.bic
    hqic = results.hqic
    llf = results.llf

    rmse = np.sqrt(np.mean(smoothed_resid**2))
    mae = np.mean(np.abs(smoothed_resid))
    bias = np.mean(smoothed_resid)

    ss_res = np.sum(smoothed_resid**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    std_res_mean = np.mean(smoothed_std_resid)
    std_res_std = np.std(smoothed_std_resid)
    # Fraction of standardized residuals falling within ±1 standard deviation.
    # For perfectly calibrated Gaussian residuals, expected ≈ 68.3%.
    inside1 = np.mean(np.abs(smoothed_std_resid) <= 1)

    # Fraction of standardized residuals falling within ±2 standard deviations.
    # For perfectly calibrated Gaussian residuals, expected ≈ 95.4%.
    inside2 = np.mean(np.abs(smoothed_std_resid) <= 2)

    # Fraction of standardized residuals falling within ±3 standard deviations.
    # For perfectly calibrated Gaussian residuals, expected ≈ 99.7%.
    inside3 = np.mean(np.abs(smoothed_std_resid) <= 3)

    lb = acorr_ljungbox(
    smoothed_std_resid,
    lags=[1, 5, 10],
    return_df=True
    )

    lb_p1  = lb.loc[1, "lb_pvalue"]
    lb_p5  = lb.loc[5, "lb_pvalue"]
    lb_p10 = lb.loc[10, "lb_pvalue"]

    jb_stat, jb_p = jarque_bera(smoothed_std_resid)
    # ---------------------------------------------------------
    # Build the report
    # ---------------------------------------------------------

    with PdfPages(path) as pdf:

        # Summary page
        add_text_page(
            pdf,
            str(summary_text),
            title="State Space Model Summary"
        )

        add_text_page(
            pdf,
            mle_text,
            title="MLE Optimization"
        )

        # ---------------------------------------------------------
        # Observation / Filtered / Smoothed comparison
        # ---------------------------------------------------------

        fig, ax = plt.subplots(figsize=(11, 6))

        x = np.arange(len(obs))

        ax.plot(
            x,
            obs,
            ".",
            color="black",
            alpha=0.5,
            label="Observations"
        )

        ax.plot(
            x,
            smoothed_state,
            color="tab:blue",
            lw=2,
            label="Smoothed state"
        )

        if meas_unc is not None:
            ax.fill_between(
                x,
                obs - meas_unc,
                obs + meas_unc,
                color="gray",
                alpha=0.2,
                label="Observation uncertainty"
            )

        # smoothed uncertainty
        ax.fill_between(
            x,
            smoothed_state - 2 * smoothed_std,
            smoothed_state + 2 * smoothed_std,
            color="tab:blue",
            alpha=0.20,
            label="Smoothed 95% CI"
        )
        if isinstance(zoom, (tuple, list, np.ndarray)) and len(zoom) == 2:
            xmin, xmax = map(int, zoom)

            ax.set_xlim(xmin, xmax)

            yvals = np.concatenate([
                obs[xmin:xmax + 1],
                smoothed_state[xmin:xmax + 1]
            ])

            if len(yvals):
                ymin = np.nanmin(yvals)
                ymax = np.nanmax(yvals)
                pad = 0.05 * (ymax - ymin if ymax > ymin else 1)

        ax.set_ylim(ymin - pad, ymax + pad)

        ax.set_title("Observations vs Smoothed State")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if plot:
            plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 8),
        sharex=True
        )

        axes[0].plot(smoothed_resid, label="Smoothed residual",
            color="tab:blue")
        axes[0].axhline(0, color="black", ls="--")
        axes[0].set_title("Residuals")
        axes[0].legend()

        axes[1].plot(smoothed_std_resid,
            color="tab:blue",
            label="Smoothed standardized residual")
        axes[1].axhline(0, color="black", ls="--")
        axes[1].axhline(2, color="red", ls=":")
        axes[1].axhline(-2, color="red", ls=":")
        axes[1].set_title("Standardized Residuals")
        axes[1].legend()
        if plot:
            plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        fit_text = f"""

        Residuals Metrics
        -------------
        RMSE                : {rmse:.4f}
            Lower is better.
            Represents the typical fitting error.

        MAE                 : {mae:.4f}
            Lower is better.
            Average absolute deviation between observations and smoothed estimates.

        Bias                : {bias:.4f}
            Measures systematic over- or under-estimation.

        R²                  : {r2:.4f}
            Fraction of observational variance reproduced by the smoothed state.
            Higher is generally better.
            Values should be interpreted in light of the measurement noise level.

        Standardize Residual Metrics
        --------------------
        Std. Residual Mean  : {std_res_mean:.4f}
            Expected value: 0.
            Large deviations suggest model bias.

        Std. Residual Std   : {std_res_std:.4f}
            Expected value: 1.
            Values >1 indicate underestimated uncertainty.
            Values <1 indicate overestimated uncertainty.


        Standardize Residual Coverage Diagnostics
        --------------------
        Within ±1σ          : {inside1:.1%}
            Theoretical Gaussian value: 68.3%

        Within ±2σ          : {inside2:.1%}
            Theoretical Gaussian value: 95.4%

        Within ±3σ          : {inside3:.1%}
            Theoretical Gaussian value: 99.7%

        Interpretation:
            Coverage significantly below expectations:
                uncertainty is underestimated.
            Coverage significantly above expectations:
                uncertainty is overestimated.

        Likelihood Metrics
        ------------------
        Log-Likelihood      : {llf:.3f}
            Higher is better when comparing models.

        AIC                 : {aic:.3f}
            Lower is better.
            Useful for comparing alternative models.

        BIC                 : {bic:.3f}
            Lower is better.
            Stronger penalty for model complexity than AIC.


        Standardize Residual Statistical Tests
        -----------------
        Ljung-Box Test (Residual Independence)
        Lag 1  p-value : {lb_p1  :.4f}
        Lag 5  p-value : {lb_p5  :.4f}
        Lag 10 p-value : {lb_p10 :.4f}

        Target:
            p > 0.05

        Interpretation:
            p > 0.05  No evidence of residual autocorrelation.
            p ≤ 0.05  Remaining temporal structure is present and
                    may not be explained by the model.

        Jarque-Bera p-value (Normality): {jb_p:.4f}
            Tests whether residuals are approximately Gaussian.
            Desired: p > 0.05

            Interpretation:
                p > 0.05  Residuals consistent with normality.
                p ≤ 0.05  Non-Gaussian residual behaviour detected.
        """

        add_text_page(
        pdf,
        fit_text,
        title="Goodness of Fit"
    )

    print(f"Saved report: {path}")
