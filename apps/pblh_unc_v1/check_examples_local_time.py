import pickle
import numpy as np
import pandas as pd
import pytz
import re
import gruanpy as gp

pids=['857603',
      '857579',
      '857947',
      '857567',
      '879125',
      '879119']

#pids = ['857947',]

SITE_TIMEZONES = {
    "LIN": "Europe/Berlin",
    "HKO": "Asia/Hong_Kong",
    "LAU": "Pacific/Auckland",
    "POT": "Europe/Rome",
}

def extract_site_code(pkl_path):
    m = re.search(r"_(LIN|HKO|LAU|POT)-", pkl_path)
    if m:
        return m.group(1)
    raise ValueError("Site code not found in pkl_path")

paths=[r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024.pkl",
    r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024.pkl",
    r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024.pkl"]

path=paths[1]
dataset=gp.read_pkl(path)

dataset={pid: dataset[pid] for pid in pids if pid in dataset}
print(f"Profiles: {len(dataset)}")

starttimes = {(pid, gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]) for pid, gdp in dataset.items()}
tods = {(pid, gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.TimeOfDay']['Value'].values[0]) for pid, gdp in dataset.items()}
print(starttimes)
print(tods)

site_code = extract_site_code(path)

for pid, gdp in dataset.items():
    # Extract UTC launch time from dataset
    launch_time_utc = pd.to_datetime(gdp.data['time'].iloc[0], utc=True)
    print(pid, launch_time_utc)
    # Convert to local timezone
    tz = pytz.timezone(SITE_TIMEZONES[site_code])
    launch_time_local = launch_time_utc.astimezone(tz)
    print(pid, launch_time_local)