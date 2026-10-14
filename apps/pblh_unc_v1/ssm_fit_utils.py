
import io
import warnings
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde, norm, probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
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
    Add documentation

    MLEResults
    Class to hold results from fitting a state space model.
    
    model : Model instance
        A reference to the model that was fit.
    filter_results : KalmanFilter instance
        The underlying state space model and Kalman filter output
    nobs : float
        The number of observations used to fit the model.
    params : ndarray
        The parameters of the model.
    scale : float
        This is currently set to 1.0 unless the model uses concentrated
        filtering.
    """
    # ---------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------
    obs = np.asarray(observations, dtype=float)
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

    mle_retvals = results.mle_retvals
    mle_settings = results.mle_settings

    summary=results.summary()

    summary_text = results.summary().as_text()
    summary_text += "\n\nNote:\n- Residuals in diagnostic tests are one-step-ahead prediction errors."
    if prior_meas_unc_flag:
        summary_text += "\n- Measurement unceratinty is fixed before MLE."
    else:
        summary_text += "\n- Measurement unceratinty is estimated by MLE."
    
    retvals_text = pprint.pformat(results.mle_retvals, indent=2)
    settings_text = pprint.pformat(results.mle_settings, indent=2)
    mle_text = retvals_text +"\n"+ settings_text

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

    print(f"Saved report: {path}")
