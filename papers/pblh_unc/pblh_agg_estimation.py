import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import gruanpy as gp
from methodology import *
from tqdm import tqdm

# Input pickle paths
pkl_paths = [
    r"papers\pblh_unc\gdp_2024_POT-RS-02_2024.pkl",
    r"papers\pblh_unc\gdp_2024_POT-RS-01_2024.pkl",
    r"papers\pblh_unc\gdp_2024_HKO-RS-01_2024.pkl",
    r"papers\pblh_unc\gdp_2024_LAU-RS-02_2024.pkl",
    r"papers\pblh_unc\gdp_2024_LIN-RS-01_2024.pkl"
]

for pkl_path in pkl_paths:

    # Load dataset
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    print("-----" * 10)
    print(f"Dataset: {pkl_path}")

    results_dict = {}

    for pid, gdp in tqdm(dataset.items()):

        # Limit to first 3.5 km
        upper_bound = gp._find_upper_bound(
            gdp.data[['alt']], upper_bound=3500, return_value=True
        )
        gdp.data = gdp.data[gdp.data['alt'] <= upper_bound]

        # Metadata
        where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
        when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
        when = when[0:10] + " " + when[11:19]
        when_day = when[0:10]
        tod = gdp.global_attrs[gdp.global_attrs['Attribute'] == "g.Measurement.TimeOfDay"]['Value'].values[0]

        data = gdp.data

        # 0. GRUAN PBLH
        std_pblh = gp.pblh_values(gp.apply_pblh_methods(data))

        # 1. Fit SSM
        import contextlib, io
        def silent_fit(gdp, method, iterations):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                return fit_ssm(gdp, method=method, iterations=iterations)

        try:
            method='powell'
            model, results = silent_fit(gdp, method, 50)
        except:
            try:
                method='lbfgs'
                model, results = silent_fit(gdp, method, 100)
            except:
                pass

        # 2. Smooth variables
        smooth_vars = smooth_variables(results.smoothed_state, results.smoothed_state_cov)

        # 3. SSM PBLH
        sm_pblh = smooth_pblh(smooth_vars)

        # 4. Monte‑Carlo simulations
        M = 200
        simulations = simulate_ssm(model, M, seed=42)

        sim_pblh_list = []
        for sim in simulations:
            sim_vars = smooth_variables(sim)
            sim_pblh = smooth_pblh(sim_vars)
            sim_pblh_list.append(sim_pblh)

        df_sim = pd.DataFrame(sim_pblh_list)

        # Uncertainty statistics
        pblh_unc = {
            key: {
                "mean": df_sim[key].mean(),
                "median": df_sim[key].median(),
                "std": df_sim[key].std(),
                "2.5": df_sim[key].quantile(0.025),
                "97.5": df_sim[key].quantile(0.975),
            }
            for key in ["pm", "thv", "rh", "ri"]
        }

        # Build pblh_info
        pblh_info = {
            "pm": {
                "value":  std_pblh[0],
                "median": pblh_unc["pm"]["median"],
                "low":    pblh_unc["pm"]["2.5"],
                "high":   pblh_unc["pm"]["97.5"],
                "samples": df_sim["pm"].values,
            },
            "thv": {
                "value":  std_pblh[1],
                "median": pblh_unc["thv"]["median"],
                "low":    pblh_unc["thv"]["2.5"],
                "high":   pblh_unc["thv"]["97.5"],
                "samples": df_sim["thv"].values,
            },
            "rh": {
                "value":  std_pblh[2],
                "median": pblh_unc["rh"]["median"],
                "low":    pblh_unc["rh"]["2.5"],
                "high":   pblh_unc["rh"]["97.5"],
                "samples": df_sim["rh"].values,
            },
            "ri": {
                "value":  std_pblh[3],
                "median": pblh_unc["ri"]["median"],
                "low":    pblh_unc["ri"]["2.5"],
                "high":   pblh_unc["ri"]["97.5"],
                "samples": df_sim["ri"].values,
            }
        }

        # Store everything INCLUDING pid
        results_dict[pid] = {
            "pid": pid,                # <--- ADDED
            "pblh_info": pblh_info,
            "where": where,
            "when": when_day,
            "tod": tod,
        }

    # -----------------------------------------
    # SAVE RESULTS FOR THIS PICKLE
    # -----------------------------------------

    base = os.path.basename(pkl_path).replace(".pkl", "_pblh_results.pkl")
    out_path = os.path.join("papers", "pblh_unc", base)

    with open(out_path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Saved: {out_path}")
