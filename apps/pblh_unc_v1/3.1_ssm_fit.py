import pickle
import gruanpy as gp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT, DeterministicLevelLLT, UnivariateLLT_AR1

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

dataset = gp.read_pkl(hko_path)


for pid, gdp in dataset.items():
    data=gdp.data
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    optimizers=['newton', 'lbfgs', 'powell']
    vars=['alt', 'theta_v', 'rh', 'wzon', 'wmeri']
    models = [UnivariateLLL, UnivariateLLT, DeterministicLevelLLT, UnivariateLLT_AR1]
    unc=[var+'_uc' for var in vars]


    opt='powell'
    var='wmeri'
    measurement_sigma2=(data[var+'_uc']*0.5)**2
    measurement_sigma2*=1
    model=DeterministicLevelLLT
    ssm=model(data[var])#, measurement_sigma2)
    results=ssm.fit(maxiter=200, method=opt, disp=True)
    
    #from statsmodels.tsa.arima.model import ARIMA
    #model = ARIMA(data[var], trend="t", order=(1, 1, 1))
    #results = model.fit()

    print(results.summary())
    plt.figure(figsize=(12,6))

    # Original data
    plt.plot(data[var], label='Observations', alpha=0.7)

    # Level component (state 0)
    plt.plot(results.smoothed_state[0], label='Smoothed Level State', linestyle='--')

    plt.title(f"{var}: original vs smoothed state")
    plt.legend()
    plt.grid(True)
    plt.show()

    fig=results.plot_diagnostics()
    fig.set_size_inches(15,10)
    plt.show(block=True)

    break
