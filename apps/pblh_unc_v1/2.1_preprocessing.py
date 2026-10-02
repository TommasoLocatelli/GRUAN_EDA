"""
This script applies preprocessing criteria on original dataset.
"""

import pickle
import pytz
from collections import Counter
import gruanpy as gp
import numpy as np
import matplotlib.pyplot as plt
from missing_data_utils import *

# ---------------------------------------------------------
# Load PKLs
# ---------------------------------------------------------

hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024.pkl"
lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024.pkl"
lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024.pkl"

hko = gp.read_pkl(hko_path)
lau = gp.read_pkl(lau_path)
lin = gp.read_pkl(lin_path)

# Original counts
orig_hko = len(hko)
orig_lau = len(lau)
orig_lin = len(lin)

# ---------------------------------------------------------
# Missing data filtering
# ---------------------------------------------------------

hko_md = [gdp.qc_results['missing_data'] for pid, gdp in hko.items()]
lau_md = [gdp.qc_results['missing_data'] for pid, gdp in lau.items()]
lin_md = [gdp.qc_results['missing_data'] for pid, gdp in lin.items()]

hko_counts = missing_count_per_profile(hko_md)
lau_counts = missing_count_per_profile(lau_md)
lin_counts = missing_count_per_profile(lin_md)

TH = 15
print(f'Missing values threshold {TH}')

# Count deletions due to missing data
del_md_hko = sum(count > TH for count in hko_counts)
del_md_lau = sum(count > TH for count in lau_counts)
del_md_lin = sum(count > TH for count in lin_counts)

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
    899535, 902160, 879420, 879500, 895881, 895978, 896317, 896285, 896461, 896506,
    900862, 900922, 900986, 900990, 922207, 922295, 922297, 922325, 922357, 922422,
    922426, 922432, 882327, 898859, 899605, 901376, 901614, 901631, 901695, 901851,
    902754, 902826, 903679, 904337, 904487, 904506, 904714, 904899, 904920, 905360,
    905503, 905637, 905718, 905843, 906011, 906364, 906607, 906981, 907115
}
bad_pids = {str(pid) for pid in bad_pids}

print("Removing profiles with alt_uc outliers...")

before_hko = len(hko)
hko = {pid: gdp for pid, gdp in hko.items() if pid not in bad_pids}
del_alt_hko = before_hko - len(hko)

before_lau = len(lau)
lau = {pid: gdp for pid, gdp in lau.items() if pid not in bad_pids}
del_alt_lau = before_lau - len(lau)

before_lin = len(lin)
lin = {pid: gdp for pid, gdp in lin.items() if pid not in bad_pids}
del_alt_lin = before_lin - len(lin)

print("Removed profiles due to alt_uc outliers:")
print(f"HKO: {del_alt_hko}")
print(f"LAU: {del_alt_lau}")
print(f"LIN: {del_alt_lin}")


# ---------------------------------------------------------
# Compute Virtual Potential Temperature
# ---------------------------------------------------------
from gruanpy.physics.formulas import virtual_potential_temperature, virtual_potential_temperature_uncertainty
import tqdm

def compute_theta(dataset):
    for pid, gdp in tqdm.tqdm(dataset.items()):
        data = gdp.data

        r_ppm     = data['wvmr_mass'].values
        r_ppm_unc = data['wvmr_mass_uc'].values

        # ppm → kg/kg
        r     = r_ppm * 1e-6
        r_unc = r_ppm_unc * 1e-6

        data['theta'] = virtual_potential_temperature(
            data['temp'], data['press'], r
        )

        data['theta_uc'] = virtual_potential_temperature_uncertainty(
            data['temp'], data['press'], r,
            data['temp_uc'], data['press_uc'], r_unc
        )

# Apply to all datasets
for dataset in [hko, lau, lin]:
    compute_theta(dataset)

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

# ---------------------------------------------------------
# Final counts
# ---------------------------------------------------------

final_hko = len(hko)
final_lau = len(lau)
final_lin = len(lin)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n================ SUMMARY ================\n")

print("Original number of profiles:")
print(f"HKO: {orig_hko}")
print(f"LAU: {orig_lau}")
print(f"LIN: {orig_lin}\n")

print("Deleted due to missing data:")
print(f"HKO: {del_md_hko}")
print(f"LAU: {del_md_lau}")
print(f"LIN: {del_md_lin}\n")

print("Deleted due to alt_uc anomalies:")
print(f"HKO: {del_alt_hko}")
print(f"LAU: {del_alt_lau}")
print(f"LIN: {del_alt_lin}\n")

print("Final number of profiles:")
print(f"HKO: {final_hko}")
print(f"LAU: {final_lau}")
print(f"LIN: {final_lin}")

print("\n=========================================\n")
