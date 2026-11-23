
import time
import pickle
import gruanpy as gp
import tqdm
import os
from datetime import datetime
import pandas as pd

if False: # prepare pkls dataset
    COLUMNS_OF_INTEREST=['time',
                        'alt', 'alt_uc', 'alt_gph_uc_ucor', 'alt_gph_uc_tcor',
                        'press', 'press_uc', 'press_gnss_uc_ucor', 'press_gnss_uc_tcor',
                        'temp', 'temp_uc', 'temp_uc_ucor', 'temp_uc_scor', 'temp_uc_tcor',
                        'rh', 'rh_uc', 'rh_uc_ucor', 'rh_uc_tcor',
                        'wvmr_mass', 'wvmr_mass_uc', 'wvmr_mass_uc_ucor', 'wvmr_mass_uc_scor', 'wvmr_mass_uc_tcor',
                        'wzon', 'wzon_uc', 'wzon_uc_ucor',
                        'wmeri', 'wmeri_uc', 'wmeri_uc_ucor'
                        ]

    # --- Logging setup ---
    LOG_DIR = "apps\pblh_unc_v1\logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"gdp_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}_unc.log")

    def log(msg):
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    # ---------------------------------------------

    start_time = time.time()

    folders = [
        r'data\products_RS41-GDP-1_HKO-RS-01_2024',
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

        for nc in tqdm.tqdm(nc_files[:3]):
            try:
                g = gp.read_gdp(nc, upper_bound=4000, columns=COLUMNS_OF_INTEREST)
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

        output_path = f"apps\\pblh_unc_v1\\pkls\\gdp_2024_{folder[24:]}_unc.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        log(f"Dataset saved to: {output_path}")

    # <-- end timer
    end_time = time.time()
    elapsed = end_time - start_time
    log(f"Total execution time: {elapsed:.2f} seconds")

    log(f"Log saved to: {log_path}")

if True: #look at unc budget
    hko_path = r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024_unc.pkl"
    lau_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024_unc.pkl"
    lin_path = r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024_unc.pkl"

    hko = gp.read_pkl(hko_path)
    lau = gp.read_pkl(lau_path)
    lin = gp.read_pkl(lin_path)

    dataset = hko | lau | lin

    data = pd.concat(
    [gdp.data for gdp in dataset.values()],
    ignore_index=True
    )


    # identify variables with total uncertainty columns
    variables = [
        c[:-3] for c in data.columns
        if c.endswith("_uc")
        and not c.endswith("_uc_ucor")
        and not c.endswith("_uc_scor")
        and not c.endswith("_uc_tcor")
    ]

    # NB from expanded to standard unc !!!

    for var in variables:

        # uncertainty components for this variable
        comps = [
            c for c in data.columns
            if c.startswith(var + "_uc_")
        ]

        if not comps:
            continue

        # variance contribution
        total_var = sum(data[c]**2 for c in comps)
        total_var = data[var+'_uc']
        print(f'Total Var: {total_var}')

        print(f"\n{var}")
        print("-" * len(var))

        for comp in comps:

            pct = (
                (data[comp]**2 / total_var)
                .mean(skipna=True)
                * 100
            )

            print(f"{comp:<30s} {pct:6.2f}%")


