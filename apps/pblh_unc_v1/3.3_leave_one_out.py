import gruanpy as gp
from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT, DeterministicLevelLLT
import numpy as np
from ssm_fit_utils import loo_metrics

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

dataset = gp.read_pkl(hko_path)

for pid, gdp in dataset.items():
    data=gdp.data
    opt='powell'
    vars=['theta_v',]#'alt', ,  'wzon']#,'rh', 'wmeri']
    model = UnivariateLLT#[UnivariateLLL, UnivariateLLT]#, DeterministicLevelLLT]
    prior_meas_unc_flag = True
    gruan_unc_coef=1
    for var in vars:
        obs = data[var]
        sigma2 = (data[var + '_uc'] * gruan_unc_coef * 0.5) ** 2

        metrics = loo_metrics(
            obs=obs,
            sigma2=sigma2,
            model=UnivariateLLT,
            step=20,
            fit_kwargs=dict(maxiter=200, method="powell", disp=False),
        )

        for k, v in metrics.items():
            if np.isscalar(v):
                print(f"{k:15s}: {v}")
        raise
    