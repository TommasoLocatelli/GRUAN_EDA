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
from applications.pblh_unc.methodology import *
from applications.pblh_unc.plot_profile import *

pkl_path = r"applications\pblh_unc\pkls\gdp_2024_LIN-RS-01_2024.pkl"
pkl_path = r'applications\pblh_unc\pkls\gdp_2024_HKO-RS-01_2024.pkl'
with open(pkl_path, "rb") as f:
    dataset = pickle.load(f)

print("Dataset Loaded")

for pid, gdp in tqdm(dataset.items()):

    print(f"\nProcessing profile: {pid}")
    # Limit altitude
    upper_bound = gp._find_upper_bound(gdp.data[['alt']], upper_bound=3500, return_value=True)
    gdp.data = gdp.data[gdp.data['alt'] <= upper_bound]

    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
    when = when[0:10] + " " + when[11:19]

    # Standard plug-in PBLH
    std_pblh = gp.pblh_values(gp.apply_pblh_methods(gdp.data))

    # Fit SSM
    mle_methods=["powell" 
                #,"lbfgs"
                #,'newton'
                #,'nm', 'bfgs', 'cg', 'ncg', 'basinhopping'
                ]

    for method in mle_methods:
        try:
            model, results = fit_ssm(gdp, method=method, iterations=100)
        except:
            print(f'MLE by {method} failed.')
        # - Print a table summarizing estimation results
        print(results.summary())

        # - Print only the estimated parameters
        print(results.params)

        # - Create diagnostic figures based on standardized residuals:
        #   (1) time series graph
        #   (2) histogram
        #   (3) Q-Q plot
        #   (4) correlogram
        for var in range(5):
            fig=results.plot_diagnostics(variable=var)
            plt.show(block=True)

        # - Examine diagnostic hypothesis tests
        # Jarque-Bera: [test_statistic, pvalue, skewness, kurtosis]
        print(results.test_normality(method='jarquebera'))
        # Goldfeld-Quandt type test: [test_statistic, pvalue]
        print(results.test_heteroskedasticity(method='breakvar'))
        # Ljung-Box test: [test_statistic, pvalue] for each lag
        print(results.test_serial_correlation(method='ljungbox'))
    break  # remove this to process all profiles
