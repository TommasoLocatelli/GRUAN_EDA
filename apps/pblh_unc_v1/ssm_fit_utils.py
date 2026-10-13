
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

def report_smoothing_goodness(results, observations, meas_unc=None, path='test.pdf'):
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

    model = results.model
    nobs = results.nobs
    params = results.params

    obs = np.asarray(observations, dtype=float)
    meas_unc = np.asarray(meas_unc, dtype=float)

    # one-step-ahead prediction residuals (innovations)
    pred_resid = results.filter_results.forecasts_error[0]

    # innovation variance
    pred_var = results.filter_results.forecasts_error_cov[0, 0, :]

    # standardized innovations
    pred_std_resid = pred_resid / np.sqrt(pred_var)

    filtered = results.states.filtered.iloc[:, 0].to_numpy()
    filtered_cov = results.states.filtered_cov
    filtered_var = results.states.filtered_cov.iloc[:, 0].to_numpy()
    filtered_std = np.sqrt(filtered_var)

    smoothed_state=results.smoothed_state[0]
    smoothed_state_cov=results.smoothed_state_cov
    smoothed_var = smoothed_state_cov[0, 0, :]
    smoothed_std = np.sqrt(smoothed_var)

    # residuals
    filtered_resid = obs - filtered
    smoothed_resid = obs - smoothed_state

    # standardized residuals
    filtered_std_resid = filtered_resid / filtered_std
    smoothed_std_resid = smoothed_resid / smoothed_std

    mle_retvals = results.mle_retvals
    mle_settings = results.mle_settings

    summary=results.summary()

    summary_text = results.summary().as_text()
    summary_text += "\n\nNote:\nResiduals in diagnostic tests are one-step-ahead prediction errors."
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
            filtered,
            color="tab:orange",
            lw=1.5,
            label="Filtered state"
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

        # filtered uncertainty
        ax.fill_between(
            x,
            filtered - 2 * filtered_std,
            filtered + 2 * filtered_std,
            color="tab:orange",
            alpha=0.20,
            label="Filtered 95% CI"
        )
        ax.set_title("Observations vs Filtered vs Smoothed State")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 8),
        sharex=True
        )

        axes[0].plot(filtered_resid, label="Filtered residual")
        axes[0].plot(smoothed_resid, label="Smoothed residual")
        axes[0].axhline(0, color="black", ls="--")
        axes[0].set_title("Residuals")
        axes[0].legend()

        axes[1].plot(filtered_std_resid,
                    label="Filtered standardized residual")
        axes[1].plot(smoothed_std_resid,
                    label="Smoothed standardized residual")
        axes[1].axhline(0, color="black", ls="--")
        axes[1].axhline(2, color="red", ls=":")
        axes[1].axhline(-2, color="red", ls=":")
        axes[1].set_title("Standardized Residuals")
        axes[1].legend()
        plt.show()

        pdf.savefig(fig)
        plt.close(fig)

        from scipy.stats import ttest_1samp
        from scipy.stats import shapiro
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.stats.stattools import jarque_bera

        pred_std_resid = pred_std_resid[np.isfinite(pred_std_resid)]

        # Mean = 0
        t_stat, t_p = ttest_1samp(pred_std_resid, 0)

        # Normality
        jb_stat, jb_p, skew, kurt = jarque_bera(pred_std_resid)

        # Optional normality test
        sw_stat, sw_p = shapiro(pred_std_resid)

        # Whiteness
        lb10 = acorr_ljungbox(pred_std_resid, lags=[10], return_df=True)
        lb20 = acorr_ljungbox(pred_std_resid, lags=[20], return_df=True)

        def verdict(p, alpha=0.05):
            return "PASS" if p > alpha else "FAIL"

        import pandas as pd

        diagnostics = pd.DataFrame([
            {
                "Test": "Mean = 0 (t-test)",
                "Statistic": t_stat,
                "p-value": t_p,
                "Decision": verdict(t_p)
            },
            {
                "Test": "Jarque-Bera Normality",
                "Statistic": jb_stat,
                "p-value": jb_p,
                "Decision": verdict(jb_p)
            },
            {
                "Test": "Shapiro-Wilk Normality",
                "Statistic": sw_stat,
                "p-value": sw_p,
                "Decision": verdict(sw_p)
            },
            {
                "Test": "Ljung-Box (lag 10)",
                "Statistic": lb10["lb_stat"].iloc[0],
                "p-value": lb10["lb_pvalue"].iloc[0],
                "Decision": verdict(lb10["lb_pvalue"].iloc[0])
            },
            {
                "Test": "Ljung-Box (lag 20)",
                "Statistic": lb20["lb_stat"].iloc[0],
                "p-value": lb20["lb_pvalue"].iloc[0],
                "Decision": verdict(lb20["lb_pvalue"].iloc[0])
            }
        ])

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axis('off')

        table = ax.table(
            cellText=np.round(
                diagnostics[["Statistic", "p-value"]].values,
                4
            ),
            colLabels=["Statistic", "p-value"],
            rowLabels=diagnostics["Test"],
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        ax.set_title(
            "Diagnostic Tests on Standardized One-Step-Ahead Prediction Residuals",
            fontsize=12,
            pad=20
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved report: {path}")
