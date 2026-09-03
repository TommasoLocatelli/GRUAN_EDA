import pickle
import gruanpy as gp
import tqdm
from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT
import numpy as np

# ---------------------------------------------------------
# Load GDPs
# ---------------------------------------------------------

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

hko = gp.read_pkl(hko_path)
lau = gp.read_pkl(lau_path)
lin = gp.read_pkl(lin_path)

gdps = {**hko, **lau, **lin}
print("Total GDPs:", len(gdps))

variables = ['alt', 'theta', 'rh', 'wmeri', 'wzon']

results = dict()

# ---------------------------------------------------------
# Fit all models
# ---------------------------------------------------------

for pid, gdp in tqdm.tqdm(gdps.items()):
    data = gdp.data
    pid_results = dict()

    for var in variables:
        endog = data[var].values.astype(float)
        endog_uc = data[var + '_uc'].values.astype(float)

        # GRUAN measurement uncertainty → measurement variance
        measurement_sigma2 = (endog_uc * 0.5)**2

        # -----------------------------
        # LLL MLE
        # -----------------------------
        lll_mle = UnivariateLLL(endog=endog)
        lll_mle_results = lll_mle.fit(
            method='powell',
            maxiter=100,
            full_output=1,
            disp=False
        )

        # -----------------------------
        # LLL GRUAN
        # -----------------------------
        lll_gruan = UnivariateLLL(endog=endog, measurement_sigma2=measurement_sigma2)
        lll_gruan_results = lll_gruan.fit(
            method='powell',
            maxiter=100,
            full_output=1,
            disp=False
        )

        # -----------------------------
        # LLT MLE
        # -----------------------------
        llt_mle = UnivariateLLT(endog=endog)
        llt_mle_results = llt_mle.fit(
            method='powell',
            maxiter=100,
            full_output=1,
            disp=False
        )

        # -----------------------------
        # LLT GRUAN
        # -----------------------------
        llt_gruan = UnivariateLLT(endog=endog, measurement_sigma2=measurement_sigma2)
        llt_gruan_results = llt_gruan.fit(
            method='powell',
            maxiter=100,
            full_output=1,
            disp=False
        )

        # Store results
        pid_results[var] = {
            'lll_mle': (lll_mle, lll_mle_results),
            'lll_gruan': (lll_gruan, lll_gruan_results),
            'llt_mle': (llt_mle, llt_mle_results),
            'llt_gruan': (llt_gruan, llt_gruan_results)
        }

    results[pid] = pid_results

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_path = "ssm_fit_lll_llt_hko_lau_lin_2024.pkl"

with open(output_path, "wb") as f:
    pickle.dump(results, f)

print(f"\nSaved results to: {output_path}")
