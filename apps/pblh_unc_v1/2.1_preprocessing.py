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

from missing_data_utils import *
hko_md = [gdp.qc_results['missing_data'] for pid, gdp in hko.items()]
lau_md = [gdp.qc_results['missing_data'] for pid, gdp in lau.items()]
lin_md = [gdp.qc_results['missing_data'] for pid, gdp in lin.items()]

hko_counts = missing_count_per_profile(hko_md)
lau_counts = missing_count_per_profile(lau_md)
lin_counts = missing_count_per_profile(lin_md)

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

# ---------------------------------------------------------
# Remove profiles with alt_uc outliers (bad PIDs)
# ---------------------------------------------------------

bad_pids = {
    899535, 902160, 879420, 879500, 895881, 895978, 896461, 900862, 900922,
    900986, 900990, 922207, 922325, 904337, 904899, 905503, 905637, 906011,
    906981
}
bad_pids = {str(pid) for pid in bad_pids}

print("Removing profiles with alt_uc outliers...")

before_hko = len(hko)
hko = {pid: gdp for pid, gdp in hko.items() if pid not in bad_pids}
removed_hko = before_hko - len(hko)

before_lau = len(lau)
lau = {pid: gdp for pid, gdp in lau.items() if pid not in bad_pids}
removed_lau = before_lau - len(lau)

before_lin = len(lin)
lin = {pid: gdp for pid, gdp in lin.items() if pid not in bad_pids}
removed_lin = before_lin - len(lin)

print("Removed profiles due to alt_uc outliers:")
print(f"HKO: {removed_hko}")
print(f"LAU: {removed_lau}")
print(f"LIN: {removed_lin}")

# ---------------------------------------------------------
# Save filtered PKLs
# ---------------------------------------------------------

def filtered_path(path):
    return path.replace(".pkl", "_preprocessed.pkl")

hko_out = filtered_path(hko_path)
lau_out = filtered_path(lau_path)
lin_out = filtered_path(lin_path)

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
