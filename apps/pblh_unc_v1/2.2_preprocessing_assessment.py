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
        data = gdp.data

        alt = data["alt"].values
        theta = data["theta"].values
        theta_uc = data["theta_uc"].values

        plt.figure(figsize=(6, 8))
        plt.plot(theta, alt, label="θ")
        plt.fill_betweenx(
            alt,
            theta - theta_uc,
            theta + theta_uc,
            alpha=0.3,
            label="θ uncertainty"
        )

        plt.xlabel("Virtual Potential Temperature θ [K]")
        plt.ylabel("Altitude [m]")
        plt.title(f"{pid} – θ profile with uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show()