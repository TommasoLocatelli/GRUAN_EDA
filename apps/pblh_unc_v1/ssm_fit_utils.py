
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
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
        measurement_unc=None,
        path='test.pdf',
        plot=True,
        zoom=False):

    # SSM model
    obs = np.asarray(observations, dtype=float)
    model = results.model
    params = results.params

    prior_meas_unc_flag = True
    if "sigma2.measurement" in params.index:
        prior_meas_unc_flag = False
        measurement_sigma2 = params["sigma2.measurement"]
        measurement_sigma = np.sqrt(measurement_sigma2)
        measurement_unc = 2 * measurement_sigma

    elif measurement_unc is not None:
        measurement_unc = np.asarray(measurement_unc, dtype=float)
        measurement_sigma = measurement_unc / 2
        measurement_sigma2 = measurement_sigma**2

    else:
        measurement_unc = 1e-8
        measurement_sigma = measurement_unc / 2
        measurement_sigma2 = measurement_sigma**2

    # Smoother
    smoother_state = results.smoothed_state[0]
    smoother_state_cov = results.smoothed_state_cov
    smoother_sigma2 = np.maximum(smoother_state_cov[0, 0, :], 0.0)
    smoother_sigma = np.sqrt(smoother_sigma2)
    smoother_unc = 2 * smoother_sigma

    # Residuals
    smoother_resid = obs - smoother_state

    # MLE
    ret = results.mle_retvals
    sett = results.mle_settings
    method = sett.get("optimizer", "unknown")
    converged = ret.get("converged", "unknown")
    iterations = ret.get("iterations", "unknown")
    fcalls = ret.get("fcalls", "unknown")
    warnflag = ret.get("warnflag", "unknown")
    fopt = ret.get("fopt", np.nan)

    # Information Criterion and LogLikelihood
    aic = results.aic
    bic = results.bic
    hqic = results.hqic
    llf = results.llf

    # Errors metrics
    rmse = np.sqrt(np.mean(smoother_resid**2))
    mae = np.mean(np.abs(smoother_resid))
    bias = np.mean(smoother_resid)

    ss_res = np.sum(smoother_resid**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    dw = durbin_watson(smoother_resid)

    lb = acorr_ljungbox(
        smoother_resid,
        lags=[1, 5, 10],
        return_df=True
    )

    lb_p1 = lb.loc[1, "lb_pvalue"]
    lb_p5 = lb.loc[5, "lb_pvalue"]
    lb_p10 = lb.loc[10, "lb_pvalue"]

    max_abs_resid = np.max(np.abs(smoother_resid))

    # Default ssm summary
    summary=results.summary()

    summary_text = results.summary().as_text()
    summary_text += "\n\nNote:\n- State Space Model Summary Residuals in diagnostic tests are one-step-ahead prediction errors. Futher analysis will consider smoothed states residuals."
    if prior_meas_unc_flag:
        summary_text += "\n- Measurement unceratinty is fixed before MLE."
    else:
        summary_text += "\n- Measurement unceratinty is estimated by MLE."


    # MLE summary
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
            smoother_state,
            color="tab:blue",
            lw=2,
            label="Smoothed state"
        )

        if measurement_unc is not None:
            ax.fill_between(
                x,
                obs - measurement_unc,
                obs + measurement_unc,
                color="gray",
                alpha=0.2,
                label="Observation uncertainty"
            )

        # smoothed uncertainty
        ax.fill_between(
            x,
            smoother_state - smoother_unc,
            smoother_state + smoother_unc,
            color="tab:blue",
            alpha=0.20,
            label="Smoothed 95% CI"
        )
        if isinstance(zoom, (tuple, list, np.ndarray)) and len(zoom) == 2:
            xmin, xmax = map(int, zoom)

            ax.set_xlim(xmin, xmax)

            yvals = np.concatenate([
                obs[xmin:xmax + 1],
                smoother_state[xmin:xmax + 1]
            ])

            if len(yvals):
                ymin = np.nanmin(yvals)
                ymax = np.nanmax(yvals)
                pad = 0.05 * (ymax - ymin if ymax > ymin else 1)

            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_title("Observations vs Smoothed State (Zoom in)")
        else:
            ax.set_title("Observations vs Smoothed State")

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if plot:
            plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(11,5)
        )

        ax.plot(
            smoother_resid,
            color="tab:blue",
            label="Residuals"
        )

        ax.axhline(
            0,
            color="black",
            ls="--"
        )

        ax.axhline(
            rmse,
            color="red",
            ls=":",
            label="±RMSE"
        )

        ax.axhline(
            -rmse,
            color="red",
            ls=":"
        )

        ax.set_title(
            "Smoother Residuals"
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Residual"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend()

        if plot:
            plt.show()

        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(11,5)
        )

        plot_acf(
            smoother_resid,
            lags=min(50, len(smoother_resid)//4),
            ax=ax,
            zero=False
        )

        ax.set_title(
            "Residual Autocorrelation Function (ACF)"
        )

        ax.set_xlabel(
            "Lag"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        if plot:
            plt.show()

        pdf.savefig(fig)
        plt.close(fig)        

        fit_text = f"""

        Residual Metrics
        ----------------

        RMSE                : {rmse:.12f}

            Lower is better.
            Represents the typical reconstruction error.

        MAE                 : {mae:.12f}

            Lower is better.
            Average absolute residual.

        Bias                : {bias:.12f}

            Expected value close to zero.
            Indicates systematic over/under estimation.

        Maximum |Residual|  : {max_abs_resid:.12f}

        R²                  : {r2:.12f}

            Fraction of variance reproduced by the smoothed state.


        Information Criteria
        --------------------

        Log-Likelihood      : {llf:.3f}

        AIC                 : {aic:.3f}

        BIC                 : {bic:.3f}

        HQIC                : {hqic:.3f}


        Residual Dependence Diagnostics
        -------------------------------

        Durbin-Watson       : {dw:.12f}

            Target ≈ 2

            DW < 2 :
                positive autocorrelation

            DW > 2 :
                negative autocorrelation


        Ljung-Box p-values

        Lag 1              : {lb_p1:.12f}

        Lag 5              : {lb_p5:.12f}

        Lag 10             : {lb_p10:.12f}

        Target:
            p > 0.05

        Interpretation:

            p > 0.05
                no significant residual autocorrelation detected

            p ≤ 0.05
                remaining temporal structure exists
                and may not be explained by the model


        Notes
        -----

        These diagnostics are computed on
        observation minus smoothed-state residuals.

        They evaluate the reconstruction quality
        and residual temporal structure.

        They are not one-step-ahead forecast errors
        and should not be interpreted as formal
        state-space innovation diagnostics.
        """

        add_text_page(
        pdf,
        fit_text,
        title="Goodness of Fit"
    )

    print(f"Saved report: {path}")
