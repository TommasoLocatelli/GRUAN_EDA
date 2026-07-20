import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
import gruanpy as gp
import matplotlib.pyplot as plt
from applications.pblh_unc.methodology import *
from tqdm import tqdm
from applications.visual_config.color_map import map_labels_to_colors
from applications.pblh_unc.plot_profile import plot_ssm_diagnostics_short, plot_ssm_diagnostics_with_violin
import pytz

pids=['857603',
      '857579',
      '857947',
      '857567',
      '879125',
      '879119']

SITE_TIMEZONES = {
    "LIN": "Europe/Berlin",
    "HKO": "Asia/Hong_Kong",
    "LAU": "Pacific/Auckland",
    "POT": "Europe/Rome",
}

import re

def extract_site_code(pkl_path):
    # Match pattern like "_LIN-" or "_HKO-" etc.
    m = re.search(r"_(LIN|HKO|LAU|POT)-", pkl_path)
    if m:
        return m.group(1)
    raise ValueError("Site code not found in pkl_path")



# ---------------------------------------------------------
# Specify the pickle file you want to load
# ---------------------------------------------------------
#pkl_path = r"papers\pblh_unc\pkls\gdp_2024_POT-RS-02_2024.pkl"

#pkl_path = r"papers\pblh_unc\pkls\gdp_2024_POT-RS-01_2024.pkl"

pkl_path=r'papers\pblh_unc\pkls\gdp_2024_HKO-RS-01_2024.pkl'

pkl_path=r'papers\pblh_unc\pkls\gdp_2024_LAU-RS-02_2024.pkl'

#pkl_path=r'papers\pblh_unc\pkls\gdp_2024_LIN-RS-01_2024.pkl'

site_code = extract_site_code(pkl_path)

# ---------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------
with open(pkl_path, "rb") as f:
    dataset = pickle.load(f)
print('Dataset Loaded')
# ---------------------------------------------------------
# PBLH Analysis
# ---------------------------------------------------------
results_dict = {}

dataset={pid: dataset[pid] for pid in pids if pid in dataset}
print(f"Profiles: {len(dataset)}")
for pid, gdp in tqdm(dataset.items()):

    print(f"\nProcessing profile: {pid}")
    upper_bound=gp._find_upper_bound(gdp.data[['alt']], upper_bound=3000, return_value=True) # find the PBLH upper bound for profile
    gdp.data = gdp.data[gdp.data['alt'] <= upper_bound]  # Limit to first 3.5 km
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0] # location
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0] # time
    when=when[0:10]+' '+when[11:19]
    launch_time_utc = pd.to_datetime(gdp.data['time'].iloc[0], utc=True)
    site_code 
    # Extract UTC launch time from dataset
    launch_time_utc = pd.to_datetime(gdp.data['time'].iloc[0], utc=True)

    # Convert to local timezone
    tz = pytz.timezone(SITE_TIMEZONES[site_code])
    launch_time_local = launch_time_utc.astimezone(tz)

    data=gdp.data

    # 0. GRUAN PBLH
    std_pblh = gp.pblh_values(gp.apply_pblh_methods(data))
    # 1. Fit SSM
    try:
        method='powell'
        model, results = fit_ssm(gdp, method=method, iterations=50)
    except:
        try:
            method='lbfgs'
            model, results = fit_ssm(gdp, method=method, iterations=100)
        except:
            pass
    print(method, results.mle_retvals)

    # 2. Smooth variables
    smooth_vars = smooth_variables(results.smoothed_state, results.smoothed_state_cov)

    # 3. SSM PBLH
    sm_pblh = smooth_pblh(smooth_vars)

    # 4. Monte‑Carlo simulations
    M = 200
    simulations = simulate_ssm(model, M, seed=42)

    sim_pblh_list = []

    for sim in simulations:
        sim_vars = smooth_variables(sim)  # uses smoothed_state only
        sim_pblh = smooth_pblh(sim_vars)
        sim_pblh_list.append(sim_pblh)

    # Convert to DataFrame (this was missing!)
    df_sim = pd.DataFrame(sim_pblh_list)

    # Compute uncertainty statistics
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

    results_dict[pid] = {
        "std_pblh": std_pblh,
        "sm_pblh": sm_pblh,
        "smooth_vars": smooth_vars,
        "sim_pblh": df_sim,
        "pblh_unc": pblh_unc,
    }

    #print(results_dict)

    # -----------------------------
    # Extract smoothed variables + uncertainties
    # -----------------------------
    alt_s        = smooth_vars["alt"]
    alt_s_unc    = smooth_vars["alt_unc"]

    thv_s        = smooth_vars["thv"]
    thv_s_unc    = smooth_vars["thv_unc"]

    rh_s         = smooth_vars["rh"]
    rh_s_unc     = smooth_vars["rh_unc"]

    u_s          = smooth_vars["u"]
    u_s_unc      = smooth_vars["u_unc"]

    v_s          = smooth_vars["v"]
    v_s_unc      = smooth_vars["v_unc"]

    # Rates of change (vertical derivatives)
    alt_rc_s     = smooth_vars["alt_rc"]
    alt_rc_s_unc = smooth_vars["alt_rc_unc"]

    thv_rc_s     = smooth_vars["thv_rc"]
    thv_rc_s_unc = smooth_vars["thv_rc_unc"]

    rh_rc_s      = smooth_vars["rh_rc"]
    rh_rc_s_unc  = smooth_vars["rh_rc_unc"]

    u_rc_s       = smooth_vars["u_rc"]
    u_rc_s_unc   = smooth_vars["u_rc_unc"]

    v_rc_s       = smooth_vars["v_rc"]
    v_rc_s_unc   = smooth_vars["v_rc_unc"]

    # Gradients
    thv_g_s      = smooth_vars["thv_grad"]
    thv_g_s_unc  = smooth_vars["thv_grad_unc"]

    rh_g_s       = smooth_vars["rh_grad"]
    rh_g_s_unc   = smooth_vars["rh_grad_unc"]

    # Richardson number
    ri_s         = smooth_vars["rich"]
    ri_s_unc     = smooth_vars["rich_unc"]


    # -----------------------------
    # Extract observed variables + uncertainties (ONLY *_uc)
    # -----------------------------

    alt_o = data["alt"].values.astype(float)
    alt_o_unc = data["alt_uc"].values.astype(float)

    temp_o = data["temp"].values.astype(float)
    temp_o_unc = data["temp_uc"].values.astype(float)

    press_o = data["press"].values.astype(float)
    press_o_unc = data["press_uc"].values.astype(float)

    wvmr_o = data["wvmr_mass"].values.astype(float) * 1e-6
    wvmr_o_unc = data["wvmr_mass_uc"].values.astype(float) * 1e-6

    rh_o  = data["rh"].values.astype(float)
    rh_o_unc = data["rh_uc"].values.astype(float)

    u_o   = data["wzon"].values.astype(float)
    u_o_unc = data["wzon_uc"].values.astype(float)

    v_o   = data["wmeri"].values.astype(float)
    v_o_unc = data["wmeri_uc"].values.astype(float)

    # Virtual potential temperature + uncertainty
    thv_o = gp.virtual_potential_temperature(temp_o, press_o, wvmr_o)
    thv_o_unc = gp.virtual_potential_temperature_uncertainty(
        temp_o, press_o, wvmr_o,
        temp_o_unc, press_o_unc, wvmr_o_unc
    )

    # Observed gradients + uncertainties
    thv_g_o = gp.finite_difference_gradient(pd.Series(thv_o), pd.Series(alt_o))
    thv_g_o_unc = gp.finite_difference_gradient_uncertainty(
    pd.Series(thv_o),
    pd.Series(alt_o),
    pd.Series(thv_o_unc),
    pd.Series(alt_o_unc))

    rh_g_o  = gp.finite_difference_gradient(pd.Series(rh_o), pd.Series(alt_o))
    rh_g_o_unc = gp.finite_difference_gradient_uncertainty(
    pd.Series(rh_o),
    pd.Series(alt_o),
    pd.Series(rh_o_unc),
    pd.Series(alt_o_unc)
    )

    # Observed Richardson number + uncertainty

    ri_o = gp.bulk_richardson_number(thv_o[0], thv_o, alt_o, u_o, v_o)
    
    ri_o_unc = gp.bulk_richardson_number_uncertainty_np(
    thv_o[0], thv_o, alt_o, u_o, v_o,
    thv_o_unc[0], thv_o_unc, alt_o_unc, u_o_unc, v_o_unc
    )

    pblh_info = {
    "pm": {
        "value":  std_pblh[0],
        "median": pblh_unc["pm"]["median"],
        "low":    pblh_unc["pm"]["2.5"],
        "high":   pblh_unc["pm"]["97.5"],
    },

    "thv": {
        "value":  std_pblh[1],
        "median": pblh_unc["thv"]["median"],
        "low":    pblh_unc["thv"]["2.5"],
        "high":   pblh_unc["thv"]["97.5"],
    },

    "rh": {
        "value":  std_pblh[2],
        "median": pblh_unc["rh"]["median"],
        "low":    pblh_unc["rh"]["2.5"],
        "high":   pblh_unc["rh"]["97.5"],
    },

    "ri": {
        "value":  std_pblh[3],
        "median": pblh_unc["ri"]["median"],
        "low":    pblh_unc["ri"]["2.5"],
        "high":   pblh_unc["ri"]["97.5"],
    }}

    # Add MC samples to pblh_info
    pblh_info["pm"]["samples"]  = df_sim["pm"].values
    pblh_info["thv"]["samples"] = df_sim["thv"].values
    pblh_info["rh"]["samples"]  = df_sim["rh"].values
    pblh_info["ri"]["samples"]  = df_sim["ri"].values

    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0] # location
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0] # time
    when=when[0:10]
    tod=gdp.global_attrs[gdp.global_attrs['Attribute'] == "g.Measurement.TimeOfDay"]['Value'].values[0] # time

    
    plot_ssm_diagnostics_with_violin(
        pid, where, when, tod, launch_time_local,
        alt_o, thv_o, rh_o, u_o, v_o,
        thv_o_unc, rh_o_unc, u_o_unc, v_o_unc,
        alt_s, thv_s, rh_s, u_s, v_s,
        thv_s_unc, rh_s_unc, u_s_unc, v_s_unc,
        simulations,
        pblh_info,
        map_labels_to_colors,
        scatter_obs=False
    )    

    comment="""
    plot_ssm_diagnostics_short(
            pid, where, when, tod,
            alt_o, thv_o, rh_o, u_o, v_o,
            thv_o_unc, rh_o_unc, u_o_unc, v_o_unc,
            alt_s, thv_s, rh_s, u_s, v_s,
            thv_s_unc, rh_s_unc, u_s_unc, v_s_unc,
            simulations,
            None,#pblh_info,
            map_labels_to_colors,
            scatter_obs=False
        )
    

    plot_ssm_diagnostics_with_hist(
        pid, where, when, tod,
        alt_o, thv_o, rh_o, u_o, v_o,
        thv_o_unc, rh_o_unc, u_o_unc, v_o_unc,
        alt_s, thv_s, rh_s, u_s, v_s,
        thv_s_unc, rh_s_unc, u_s_unc, v_s_unc,
        simulations,
        pblh_info,
        map_labels_to_colors,
        scatter_obs=False
    )


    plot_ssm_diagnostics_f
    ull(
    pid,
    alt_o, thv_o, rh_o, u_o, v_o,
    thv_g_o, rh_g_o, ri_o,
    alt_o_unc, thv_o_unc, rh_o_unc, u_o_unc, v_o_unc,
    thv_g_o_unc, rh_g_o_unc, ri_o_unc,
    alt_s, thv_s, rh_s, u_s, v_s,
    thv_g_s, rh_g_s, ri_s,
    alt_s_unc, thv_s_unc, rh_s_unc, u_s_unc, v_s_unc,
    thv_g_s_unc, rh_g_s_unc, ri_s_unc,
    simulations,
    pblh_info,
    map_labels_to_colors,
    FONTE_SIZE=8,
    alpha=0.75,
    alpha_unc=0.30,
    alpha_sim=0.02
    )"""