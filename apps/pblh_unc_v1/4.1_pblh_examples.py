import gruanpy as gp
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL, MultivariateLLT, MultivariateLLTDetLevel

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

dataset = gp.read_pkl(hko_path)

variables_labels=['alt', 'theta_v',  'wzon','rh', 'wmeri']
uncertainty_labels=[var+'_uc' for var in variables_labels]

model = MultivariateLLTDetLevel
opt = 'powell'
for pid, gdp in dataset.items():
    obs = gdp.data[variables_labels]
    uncs = gdp.data[uncertainty_labels]
    sigma = uncs*0.5
    sigma2 = sigma**2
    ssm = model(obs, sigma2)
    results=ssm.fit(maxiter=200, method=opt, disp=True)
    break
