import gruanpy as gp
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel
import numpy as np

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

dataset = gp.read_pkl(hko_path)

variables_labels=['alt', 'theta_v',  'wzon','rh', 'wmeri']
uncertainty_labels=[var+'_uc' for var in variables_labels]

model = MultivariateLLT
opt = 'powell'
for pid, gdp in dataset.items():
    obs = gdp.data[variables_labels]
    uncs = gdp.data[uncertainty_labels]
    sigma = uncs*0.5
    sigma2 = sigma**2
    ssm = model(obs, sigma2)
    results=ssm.fit(maxiter=200, method=opt, disp=True)
    smoothed_state=results.smoothed_state
    smoothed_state_cov=results.smoothed_state_cov
    smoothed_state_var = np.diagonal(
        smoothed_state_cov,
        axis1=0, axis2=1).T
    smoothed_state_sigma = np.sqrt(smoothed_state_var)
    smoothe_state_unc = smoothed_state_sigma*2
    print(smoothed_state.shape, smoothed_state_var.shape)
    if smoothed_state.shape[0]==10:
        smoothed_levels=smoothed_state[0:10:2]
        smoothed_trends=smoothed_state[1:11:2]
    break
