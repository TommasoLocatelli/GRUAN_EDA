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

with open(pkl_path, "rb") as f:
    dataset = pickle.load(f)

print("Dataset Loaded")

for pid, gdp in tqdm(dataset.items()):

    print(f"\nProcessing profile: {pid}")
    if str(pid)!='857535':

        # Limit altitude
        upper_bound = gp._find_upper_bound(gdp.data[['alt']], upper_bound=3500, return_value=True)
        gdp.data = gdp.data[gdp.data['alt'] <= upper_bound]

        where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
        when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
        when = when[0:10] + " " + when[11:19]

        # Standard plug-in PBLH
        std_pblh = gp.pblh_values(gp.apply_pblh_methods(gdp.data))

        # Fit SSM
        mle_methods=["powell", 
                    "lbfgs",
                    'newton',
                    'nm',
                    'bfgs',
                    'cg',
                    'ncg',
                    'basinhopping']

        for method in mle_methods:
            try:
                model, results = fit_ssm(gdp, method=method, iterations=100)
                print(results.summary())
                results.plot_disgnostics()
                break
            except:
                print(f'MLE by {method} failed.')
        break  # remove this to process all profiles
