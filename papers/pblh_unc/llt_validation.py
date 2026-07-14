import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm

import gruanpy as gp
from methodology import *
from plot_profile import (
    plot_ssm_diagnostics_short,
    plot_ssm_diagnostics_full,
    plot_ssm_diagnostics_with_hist
)

# =====================================================================
# FULL POST-ESTIMATION VALIDATION FUNCTION
# =====================================================================

def validate_ssm_fit(model, results, gdp, profile_id=None):
    """
    Full post-estimation validation for LLT fitting.
    Prints diagnostics and produces plots.
    """

    print("\n" + "="*80)
    print(f"POST-ESTIMATION VALIDATION FOR PROFILE {profile_id}")
    print("="*80)

    # -----------------------------------------------------
    # 1. PARAMETER TABLE
    # -----------------------------------------------------
    print("\nEstimated parameters (σ²):")
    param_table = pd.DataFrame({
        "parameter": results.param_names,
        "value": results.params
    })
    print(param_table)

    # -----------------------------------------------------
    # 2. INITIAL STATE & PRECISION
    # -----------------------------------------------------
    print("\nInitial state mean:")
    print(results.filter_results.initial_state)

    print("\nInitial state covariance:")
    print(results.filter_results.initial_state_cov)

    print("\nInitial state precision (inverse covariance):")
    print(np.linalg.inv(results.filter_results.initial_state_cov))


    # -----------------------------------------------------
    # 4. RESIDUALS & FORECAST ERRORS
    # -----------------------------------------------------
    print("\nOne-step-ahead residuals (head):")
    print(results.resid)

    std_err = results.filter_results.standardized_forecasts_error
    print("\nStandardized forecast errors shape:", std_err.shape)

    # Plot standardized errors for each variable
    fig, axes = plt.subplots(std_err.shape[0], 1, figsize=(8, 12), sharex=True)
    for i in range(std_err.shape[0]):
        axes[i].plot(std_err[i], label=f"std forecast error var {i}")
        axes[i].legend()
    plt.suptitle("Standardized Forecast Errors")
    plt.show()

    # -----------------------------------------------------
    # 5. BUILT-IN DIAGNOSTIC TESTS
    # -----------------------------------------------------
    print("\nNormality test:")
    print(results.test_normality())

    print("\nSerial correlation test:")
    print(results.test_serial_correlation())

    print("\nHeteroskedasticity test:")
    print(results.test_heteroskedasticity())

    # -----------------------------------------------------
    # 6. INFORMATION CRITERIA
    # -----------------------------------------------------
    print("\nInformation criteria:")
    print(f"AIC  : {results.aic}")
    print(f"BIC  : {results.bic}")
    print(f"HQIC : {results.hqic}")
    print(f"AICC : {results.aicc}")

    # -----------------------------------------------------
    # 7. FITTED VALUES
    # -----------------------------------------------------
    print("\nFitted values (one-step-ahead predictions):")
    print(results.fittedvalues.head())

    # Plot fitted vs observed θv
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gdp.data["alt"], gdp.data["th"], ".", alpha=0.4, label="Observed θv")
    ax.plot(gdp.data["alt"], results.fittedvalues.iloc[:, 1], "-", label="Fitted θv")
    ax.set_title("Observed vs Fitted θv")
    ax.set_xlabel("Altitude")
    ax.set_ylabel("θv")
    ax.legend()
    plt.show()

    # -----------------------------------------------------
    # 8. PREDICTION & FORECAST
    # -----------------------------------------------------
    pred = results.get_prediction()
    print("\nPrediction summary frame:")
    print(pred.summary_frame().head())

    # -----------------------------------------------------
    # 9. KALMAN FILTER DIAGNOSTICS
    # -----------------------------------------------------
    print("\nKalman filter loglikelihood:", results.filter_results.llf)
    print("Kalman filter convergence:", results.filter_results.converged)

    print("\n" + "="*80)
    print("END OF VALIDATION")
    print("="*80)


# =====================================================================
# LOAD DATASET
# =====================================================================

pkl_path = r"papers\pblh_unc\gdp_2024_POT-RS-02_2024.pkl"

with open(pkl_path, "rb") as f:
    dataset = pickle.load(f)

print("Dataset Loaded")

# =====================================================================
# PROCESS EACH PROFILE
# =====================================================================

for pid, gdp in tqdm(dataset.items()):

    print(f"\nProcessing profile: {pid}")

    # Limit altitude
    upper_bound = gp._find_upper_bound(gdp.data[['alt']], upper_bound=3500, return_value=True)
    gdp.data = gdp.data[gdp.data['alt'] <= upper_bound]

    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
    when = when[0:10] + " " + when[11:19]

    # GRUAN PBLH
    std_pblh = gp.pblh_values(gp.apply_pblh_methods(gdp.data))

    # Fit SSM
    try:
        method = "powell"
        model, results = fit_ssm(gdp, method=method, iterations=50)
    except:
        try:
            method = "lbfgs"
            model, results = fit_ssm(gdp, method=method, iterations=100)
        except:
            print("Fit failed.")
            continue

    print(method, results.mle_retvals)
    print("Final loglik:", results.llf)

    # Parameter table
    param_table = pd.DataFrame({
        "parameter": results.param_names,
        "value": results.params
    })
    print(param_table)

    # FULL VALIDATION
    validate_ssm_fit(model, results, gdp, profile_id=pid)

    break  # remove this to process all profiles
