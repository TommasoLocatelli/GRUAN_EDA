import gruanpy as gp
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel
import numpy as np
import pandas as pd

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

dataset = gp.read_pkl(hko_path)

variables_labels=['alt', 'theta_v', 'rh', 'wzon', 'wmeri']
uncertainties_labels=[var+'_uc' for var in variables_labels]
labels=variables_labels+uncertainties_labels
models_classes=[MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel]
optimizers=['powell', 'lbfgs', 'nm']
M=3
seed=42
for pid, gdp in dataset.items():
    data=gdp.data[labels]
    for model in models_classes:
        for opt in optimizers:
            ssm=model(data[variables_labels], data[uncertainties_labels])
            results=ssm.fit(maxiter=200, method=opt, disp=False)
            
            mle_retvals = results.mle_retvals
            mle_settings = results.mle_settings
            summary_tables = results.summary().tables
            smoothed_state = results.smoothed_state
            smoother_state_cov = results.smoothed_state_cov
            
            simulator=ssm.simulation_smoother(seed=seed)
            simulations=[]
            for _ in range(M):
                simulator.simulate()
                simulated_state=simulator.simulated_state
                simulations.append(simulated_state)
            print(simulations[0])
            break
        break
    break
    