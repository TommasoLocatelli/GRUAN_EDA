"""
This script apply preprocessing criterion on original dataset.
"""

import pickle
import pytz
from collections import Counter
import gruanpy as gp
import numpy as np
import matplotlib.pyplot as plt

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024.pkl"


hko = gp.read_pkl(hko_path)
lau = gp.read_pkl(lau_path)
lin = gp.read_pkl(lin_path)

if True: # filter profiles with more than 10 missing data
    from missing_data_utils import *
    hko_md=[gdp.qc_results['missing_data'] for pid, gdp in hko.items()]
    lau_md=[gdp.qc_results['missing_data'] for pid, gdp in lau.items()]
    lin_md=[gdp.qc_results['missing_data'] for pid, gdp in lin.items()]
    hko_counts = missing_count_per_profile(hko_md)
    lau_counts = missing_count_per_profile(lau_md)
    lin_counts = missing_count_per_profile(lin_md)
    # Threshold
    TH = 10
    print(f'Missing values threshold {TH}')
    # Filter HKO
    hko = {
        pid: gdp
        for (pid, gdp), count in zip(hko.items(), hko_counts)
        if count <= TH
    }
    # Filter LAU
    lau = {
        pid: gdp
        for (pid, gdp), count in zip(lau.items(), lau_counts)
        if count <= TH
    }
    # Filter LIN
    lin = {
        pid: gdp
        for (pid, gdp), count in zip(lin.items(), lin_counts)
        if count <= TH
    }

    def filtered_path(path):
        return path.replace(".pkl", "_md_filtered.pkl")

    hko_out = filtered_path(hko_path)
    lau_out = filtered_path(lau_path)
    lin_out = filtered_path(lin_path)

    import pickle

    with open(hko_out, "wb") as f:
        pickle.dump(hko, f)

    with open(lau_out, "wb") as f:
        pickle.dump(lau, f)

    with open(lin_out, "wb") as f:
        pickle.dump(lin, f)

    print("Filtered PKLs saved:")
    print(hko_out)
    print(lau_out)
    print(lin_out)

