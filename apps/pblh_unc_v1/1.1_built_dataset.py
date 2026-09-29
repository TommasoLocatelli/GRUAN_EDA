"""
This script reads GDPs and builds a pickle containing them for each site.
Logs are saved in the logs/ subfolder.
"""

import time
import pickle
import gruanpy as gp
import tqdm
import os
from datetime import datetime

# --- Logging setup ---
LOG_DIR = "apps\pblh_unc_v1\logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, f"gdp_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(msg):
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(msg + "\n")

# ---------------------------------------------

start_time = time.time()

folders = [
    #r'data\products_RS41-GDP-1_HKO-RS-01_2024'#,
    r'data\products_RS41-GDP-1_LAU-RS-02_2024',
    r'data\products_RS41-GDP-1_LIN-RS-01_2024'
]

for folder in folders:
    dataset = {}

    if not os.path.isdir(folder):
        log(f"Warning: folder not found -> {folder}")
        continue

    nc_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nc")
    ]

    log(f"{folder}: {len(nc_files)} .nc files")

    if len(nc_files) == 0:
        log(f"No .nc files found in {folder}")
        continue

    if folder == r'data\products_RS41-GDP-1_LIN-RS-01_2024':
        nc_files = [
            f for f in nc_files
            if ("T000000" in f or "T120000" in f)
        ]
        log(f"Keeping only T00 and T12 files: {len(nc_files)} found")

    for nc in tqdm.tqdm(nc_files[:]):
        try:
            g = gp.read_gdp(nc, upper_bound=4000, columns=gp.COLUMNS_OF_INTEREST)
            pid = g.global_attrs[g.global_attrs['Attribute'] == 'g.Product.Id']['Value'].values[0]
            tod = gp.get_time_of_day(g)

            if tod == 'twilight':
                log(f"Skip twilight, Product.Id detected: {pid} (skipping {nc})")
                continue

            if pid in dataset:
                log(f"Duplicate Product.Id detected: {pid} (skipping {nc})")
                continue

            dataset[pid] = g

        except Exception as e:
            log(f"Error reading {nc}: {e}")

    log(f"Total unique profiles loaded: {len(dataset)}")

    output_path = f"apps\\pblh_unc_v1\\pkls\\gdp_2024_{folder[24:]}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    log(f"Dataset saved to: {output_path}")

# <-- end timer
end_time = time.time()
elapsed = end_time - start_time
log(f"Total execution time: {elapsed:.2f} seconds")

log(f"Log saved to: {log_path}")
