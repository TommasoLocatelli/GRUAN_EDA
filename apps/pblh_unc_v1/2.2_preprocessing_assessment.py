"""
This script checks preprocessing criterion on original dataset.
Need to look at Potential Virtual Temperature (theta) and its uncertainty (theta_uc)
"""

import pickle
import gruanpy as gp
import numpy as np
import matplotlib.pyplot as plt

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_preprocessed.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_preprocessed.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_preprocessed.pkl"

hko = gp.read_pkl(hko_path)
lau = gp.read_pkl(lau_path)
lin = gp.read_pkl(lin_path)

from gruanpy.gdp.quality_check import detect_outliers
import pandas as pd
import tqdm

datasets = [hko, lau, lin]
all_outliers = []

for dataset in datasets:
    for pid, gdp in tqdm.tqdm(dataset.items()):
        outliers = detect_outliers(gdp.data[['theta', 'theta_uc']], iqr_factor=100)
        
        if isinstance(outliers, pd.DataFrame) and not outliers.empty:

            # Expand each row: one row per outlier index
            expanded_rows = []

            for _, row in outliers.iterrows():
                variable = row["variable"]
                indices = row["indices"]

                for idx in indices:
                    expanded_rows.append({
                        "pid": pid,
                        "variable": variable,
                        "index": idx,
                        "value": gdp.data.loc[idx, variable]
                    })

            expanded_df = pd.DataFrame(expanded_rows)
            all_outliers.append(expanded_df)


# Combine all outlier rows into one DataFrame
if all_outliers:
    outliers_df = pd.concat(all_outliers, ignore_index=True)

    # Max theta outlier
    max_theta = outliers_df[outliers_df["variable"] == "theta"]["value"].max()

    # Max theta_uc outlier
    max_theta_uc = outliers_df[outliers_df["variable"] == "theta_uc"]["value"].max()

    print("Max theta outlier:", max_theta)
    print("Max theta_uc outlier:", max_theta_uc)
else:
    print("No outliers found.")
