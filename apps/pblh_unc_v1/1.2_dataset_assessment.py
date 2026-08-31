import pickle
import pytz
from collections import Counter
import gruanpy as gp
import numpy as np
import matplotlib.pyplot as plt

hko = gp.read_pkl(r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024.pkl")
lau = gp.read_pkl(r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024.pkl")
lin = gp.read_pkl(r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024.pkl")

print("Loaded:")
print("HKO:", len(hko))
print("LAU:", len(lau))
print("LIN:", len(lin))

"""
Goals:
- summary table
- missing data
- physics constraints
- outliers
- altitude drops
- discard some profiles?
"""