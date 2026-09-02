"""
This script analyze dataset aggregate quality check properties.
"""

import pickle
import pytz
from collections import Counter
import gruanpy as gp
import numpy as np
import matplotlib.pyplot as plt

TEXT_SIZE=22
#c="""
plt.rcParams.update({
    #"font.size": 5,            # Base font size
    "axes.titlesize": TEXT_SIZE,       # Subplot titles
    "axes.labelsize": TEXT_SIZE,       # Axis labels
    "xtick.labelsize": TEXT_SIZE,      # Tick labels
    "ytick.labelsize": TEXT_SIZE,
    "legend.fontsize": TEXT_SIZE,      # Legend text
    "figure.titlesize": TEXT_SIZE,     # Suptitle
})

hko = gp.read_pkl(r"apps\pblh_unc_v1\pkls\gdp_2024__HKO-RS-01_2024.pkl")
lau = gp.read_pkl(r"apps\pblh_unc_v1\pkls\gdp_2024__LAU-RS-02_2024.pkl")
lin = gp.read_pkl(r"apps\pblh_unc_v1\pkls\gdp_2024__LIN-RS-01_2024.pkl")

if True: # filter profiles with more than TH missing data
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

if True: # Remove profile with alt_uc outliers
    bad_pids = {
        899535, 902160, 879420, 879500, 895881, 895978, 896461, 900862, 900922,
        900986, 900990, 922207, 922325, 904337, 904899, 905503, 905637, 906011,
        906981
    }
    bad_pids={str(pid) for pid in bad_pids}

    # --- HKO ---
    before_hko = len(hko)
    hko = {
        pid: gdp
        for (pid, gdp), count in zip(hko.items(), hko_counts)
        if pid not in bad_pids
    }
    removed_hko = before_hko - len(hko)

    # --- LAU ---
    before_lau = len(lau)
    lau = {
        pid: gdp
        for (pid, gdp), count in zip(lau.items(), lau_counts)
        if pid not in bad_pids
    }
    removed_lau = before_lau - len(lau)

    # --- LIN ---
    before_lin = len(lin)
    lin = {
        pid: gdp
        for (pid, gdp), count in zip(lin.items(), lin_counts)
        if pid not in bad_pids
    }
    removed_lin = before_lin - len(lin)

    # Print summary
    print("Removed profiles:")
    print(f"HKO: {removed_hko}")
    print(f"LAU: {removed_lau}")
    print(f"LIN: {removed_lin}")

def count_nonempty(df_list):
    return sum([not df.empty for df in df_list])

if False: # look at alt drops
    # HKO
    hko_drops = [gdp.qc_results["altitude_drops"] for pid, gdp in hko.items()]
    hko_count = count_nonempty(hko_drops)

    # LAU
    lau_drops = [gdp.qc_results["altitude_drops"] for pid, gdp in lau.items()]
    lau_count = count_nonempty(lau_drops)

    # LIN
    lin_drops = [gdp.qc_results["altitude_drops"] for pid, gdp in lin.items()]
    lin_count = count_nonempty(lin_drops)

    print("Altitude drop profiles:")
    print(f"HKO: {hko_count}")
    print(f"LAU: {lau_count}")
    print(f"LIN: {lin_count}")

    def get_drop_counts(dataset):
        return pd.Series({
            pid: len(gdp.qc_results["altitude_drops"])
            for pid, gdp in dataset.items()
        })

    def get_drop_locations(dataset):
        locs = {}
        for pid, gdp in dataset.items():
            df = gdp.qc_results["altitude_drops"]
            if df.empty:
                continue
            # df["indices"] is a list of lists → flatten it
            all_indices = []
            for lst in df["indices"]:
                all_indices.extend(lst)
            locs[pid] = all_indices
        return locs

    hko_counts = get_drop_counts(hko)
    lau_counts = get_drop_counts(lau)
    lin_counts = get_drop_counts(lin)

    hko_locs = get_drop_locations(hko)
    lau_locs = get_drop_locations(lau)
    lin_locs = get_drop_locations(lin)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.hist(hko_counts, bins=range(0, hko_counts.max()+2))
    plt.title("HKO – Number of altitude drops")
    plt.xlabel("Drops per profile")
    plt.ylabel("Count")

    plt.subplot(1,3,2)
    plt.hist(lau_counts, bins=range(0, lau_counts.max()+2))
    plt.title("LAU – Number of altitude drops")
    plt.xlabel("Drops per profile")

    plt.subplot(1,3,3)
    plt.hist(lin_counts, bins=range(0, lin_counts.max()+2))
    plt.title("LIN – Number of altitude drops")
    plt.xlabel("Drops per profile")

    plt.tight_layout()
    plt.show()

    def flatten_locations(loc_dict):
        return np.array([idx for locs in loc_dict.values() for idx in locs])

    hko_loc_flat = flatten_locations(hko_locs)
    lau_loc_flat = flatten_locations(lau_locs)
    lin_loc_flat = flatten_locations(lin_locs)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.hist(hko_loc_flat, bins=50)
    plt.title("HKO – Drop locations")
    plt.xlabel("Index")
    plt.ylabel("Frequency")

    plt.subplot(1,3,2)
    plt.hist(lau_loc_flat, bins=50)
    plt.title("LAU – Drop locations")
    plt.xlabel("Index")

    plt.subplot(1,3,3)
    plt.hist(lin_loc_flat, bins=50)
    plt.title("LIN – Drop locations")
    plt.xlabel("Index")

    plt.tight_layout()
    plt.show()

if False: # check for outliers
    hko_out = [gdp.qc_results["detect_outliers"] for pid, gdp in hko.items()]
    lau_out = [gdp.qc_results["detect_outliers"] for pid, gdp in lau.items()]
    lin_out = [gdp.qc_results["detect_outliers"] for pid, gdp in lin.items()]
    print("HKO profiles with outliers:", count_nonempty(hko_out))
    print("LAU profiles with outliers:", count_nonempty(lau_out))
    print("LIN profiles with outliers:", count_nonempty(lin_out))

    def collect_outliers_with_values(site_dict, out_list, site_name):
        """
        Collect outlier QC rows and add actual variable values at the given indices.
        No filtering, no exclusions.
        """
        out = []

        for (pid, gdp), outdf in zip(site_dict.items(), out_list):
            if outdf is None or outdf.empty:
                continue

            enriched = outdf.copy()
            values_list = []

            for _, row in outdf.iterrows():
                var = row["variable"]
                idxs = row["indices"]

                # Extract actual variable values
                try:
                    vals = [gdp.data[var][i] for i in idxs]
                except Exception:
                    vals = ["N/A"] * len(idxs)

                values_list.append(vals)

            enriched["values"] = values_list

            out.append(f"{site_name} | PID: {pid}\n{enriched}\n")

        return out

    hko_outliers_txt = collect_outliers_with_values(hko, hko_out, "HKO")
    lau_outliers_txt = collect_outliers_with_values(lau, lau_out, "LAU")
    lin_outliers_txt = collect_outliers_with_values(lin, lin_out, "LIN")

    output_path = "outliers_with_values_all_sites.txt"

    with open(output_path, "w") as f:
        for entry in (hko_outliers_txt + lau_outliers_txt + lin_outliers_txt):
            f.write(entry)
            f.write("\n" + "-"*80 + "\n")

    print("Outliers with values saved to:", output_path)

if False: # calibrate outliers IQR coef
    from gruanpy.gdp.quality_check import detect_outliers
    import pandas as pd
    import tqdm

    datasets = [hko, lau, lin]
    all_outliers = []

    for dataset in datasets:
        for pid, gdp in tqdm.tqdm(dataset.items()):
            outliers = detect_outliers(gdp.data, iqr_factor=100)

            if isinstance(outliers, pd.DataFrame) and not outliers.empty:

                # Expand each row: one row per outlier index
                expanded_rows = []

                for _, row in outliers.iterrows():
                    variable = row["variable"]
                    indices = row["indices"]

                    for idx in indices:
                        expanded_rows.append({
                            "pid": pid,
                            "variable": variable,
                            "index": idx,
                            "value": gdp.data.loc[idx, variable]
                        })

                expanded_df = pd.DataFrame(expanded_rows)
                all_outliers.append(expanded_df)

    # Save final output
    if all_outliers:
        final_outliers = pd.concat(all_outliers, ignore_index=True)
        final_outliers.to_csv("outliers.txt", sep="\t", index=False)
    else:
        print("No outliers found.")

if False: # look at outliers detected

    # Read the outlier file
    df = pd.read_csv("outliers.txt", sep="\t")

    # Compute max value per variable
    max_values = df.groupby("variable")["value"].max()

    print(max_values)

    # Filter for alt_uc values greater than 100
    alt_high = df[(df["variable"] == "alt_uc") & (df["value"] > 100)]

    print(alt_high)

    print(alt_high["pid"].unique())

if False: #check physical contrainst
    hko_pc=[gdp.qc_results['physics_constraint'] for pid, gdp in hko.items()]
    lau_pc=[gdp.qc_results['physics_constraint'] for pid, gdp in lau.items()]
    lin_pc=[gdp.qc_results['physics_constraint'] for pid, gdp in lin.items()]
    print("HKO physics violations:", count_nonempty(hko_pc))
    print("LAU physics violations:", count_nonempty(lau_pc))
    print("LIN physics violations:", count_nonempty(lin_pc))

    def collect_violations(site_dict, pc_list, site_name):
        """
        Returns list of strings describing physics violations for a site,
        including actual values and altitude at violating indices,
        EXCLUDING independently:
            - wvmr_mass values between 20000 and 25000
            - altitudes between -20 and 0
        """
        out = []

        for (pid, gdp), pc in zip(site_dict.items(), pc_list):
            if pc.empty:
                continue

            pc_enriched = pc.copy()
            values_list = []
            alt_list = []
            keep_mask = []

            for _, row in pc.iterrows():
                var = row["variable"]
                idxs = row["indices"]

                # Extract actual values
                try:
                    vals = [gdp.data[var][i] for i in idxs]
                except Exception:
                    vals = ["N/A"] * len(idxs)

                # Extract altitude
                try:
                    alts = [gdp.data["alt"][i] for i in idxs]
                except Exception:
                    alts = ["N/A"] * len(idxs)

                # Independent exclusion rules
                exclude = False

                # Rule 1: wvmr_mass values between 20000–25000
                if var == "wvmr_mass":
                    if any(20000 <= v <= 25000 for v in vals):
                        exclude = True

                # Rule 2: altitude between –20–0
                if any(-20 <= a <= 0 for a in alts):
                    exclude = True

                keep_mask.append(not exclude)
                values_list.append(vals)
                alt_list.append(alts)

            # Filter out excluded rows
            pc_enriched["values"] = values_list
            pc_enriched["alt"] = alt_list
            pc_enriched = pc_enriched[keep_mask]

            if not pc_enriched.empty:
                out.append(f"{site_name} | PID: {pid}\n{pc_enriched}\n")

        return out


    violations = []
    violations += collect_violations(hko, hko_pc, "HKO")
    violations += collect_violations(lau, lau_pc, "LAU")
    violations += collect_violations(lin, lin_pc, "LIN")

    output_path = r"physics_violations_all_sites.txt"
    with open(output_path, "w") as f:
        for entry in violations:
            f.write(entry)
            f.write("\n" + "-"*80 + "\n")

    print(f"Physics violations saved to: {output_path}")

if False: # check missing data
    
    from missing_data_utils import *
    hko_md=[gdp.qc_results['missing_data'] for pid, gdp in hko.items()]
    lau_md=[gdp.qc_results['missing_data'] for pid, gdp in lau.items()]
    lin_md=[gdp.qc_results['missing_data'] for pid, gdp in lin.items()]
    print(f"HKO profiles with missing: {count_profiles_with_missing(hko_md)} over {len(hko_md)}")
    print(f"LAU profiles with missing: {count_profiles_with_missing(lau_md)} over {len(lau_md)}")
    print(f"LIN profiles with missing: {count_profiles_with_missing(lin_md)} over {len(lin_md)}")

    hko_counts = missing_count_per_profile(hko_md)
    lau_counts = missing_count_per_profile(lau_md)
    lin_counts = missing_count_per_profile(lin_md)

    # Choose a common range across all stations
    min_pos = min(min(hko_counts), min(lau_counts), min(lin_counts))
    max_pos = max(max(hko_counts), max(lau_counts), max(lin_counts))

    # Define fixed bin edges
    bins = np.linspace(min_pos, max_pos, 10)   

    plt.figure(figsize=(10, 8))
    plt.hist(hko_counts, bins=bins, alpha=0.5, label="HKO")
    plt.hist(lau_counts, bins=bins, alpha=0.5, label="LAU")
    plt.hist(lin_counts, bins=bins, alpha=0.5, label="LIN")

    plt.title("Missing Value Count per Profile – All Stations")
    plt.xlabel("Missing values")
    plt.ylabel("Profiles")
    plt.legend()
    plt.show()

    hko_gap_sizes = gap_sizes(hko_md)
    lau_gap_sizes = gap_sizes(lau_md)
    lin_gap_sizes = gap_sizes(lin_md)

    # Choose a common range across all stations
    min_pos = min(min(hko_gap_sizes), min(lau_gap_sizes), min(lin_gap_sizes))
    max_pos = max(max(hko_gap_sizes), max(lau_gap_sizes), max(lin_gap_sizes))

    # Define fixed bin edges
    bins = np.linspace(min_pos, max_pos, 7)   

    plt.figure(figsize=(10, 8))
    plt.hist(hko_gap_sizes, bins=bins, alpha=0.5, label="HKO")
    plt.hist(lau_gap_sizes, bins=bins, alpha=0.5, label="LAU")
    plt.hist(lin_gap_sizes, bins=bins, alpha=0.5, label="LIN")

    plt.title("Gap Size Distribution – All Stations")
    plt.xlabel("Gap length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    hko_positions = gap_positions(hko_md)
    lau_positions = gap_positions(lau_md)
    lin_positions = gap_positions(lin_md)

    # Choose a common range across all stations
    min_pos = min(min(hko_positions), min(lau_positions), min(lin_positions))
    max_pos = max(max(hko_positions), max(lau_positions), max(lin_positions))

    # Define fixed bin edges
    bins = np.linspace(min_pos, max_pos, 361)   

    plt.figure(figsize=(10, 8))

    plt.hist(hko_positions, bins=bins, alpha=0.5, label="HKO")
    plt.hist(lau_positions, bins=bins, alpha=0.5, label="LAU")
    plt.hist(lin_positions, bins=bins, alpha=0.5, label="LIN")

    plt.title("Missing Value Position Distribution – All Stations")
    plt.xlabel("Index (approx altitude level)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.show()

    hko_var = aggregate_missing_by_variable(hko_md)
    lau_var = aggregate_missing_by_variable(lau_md)
    lin_var = aggregate_missing_by_variable(lin_md)

    combined_missing = combine_sites(hko_var, lau_var, lin_var)

    plot_combined_missing_pie(combined_missing)

if True: # summary plots
        
    def analyze_site(dataset, site_key):
        day_night = Counter()
        seasons = Counter()
        hours = Counter()   # <-- NEW: count launches per local hour

        for pid, gdp in dataset.items():
            utc = gp.get_utc_time(gdp)
            local = gp.to_local_time(utc, site_key)

            dn = gp.get_time_of_day(gdp)
            ss = gp.classify_season(local, site_key)
            hh = local.hour  # <-- local hour (0–23)

            day_night[dn] += 1
            seasons[ss] += 1
            hours[hh] += 1

        return day_night, seasons, hours

    hko_dn, hko_season, hko_hours = analyze_site(hko, "HKO")
    lau_dn, lau_season, lau_hours = analyze_site(lau, "LAU")
    lin_dn, lin_season, lin_hours = analyze_site(lin, "LIN")


    def prepare_stacked(counter_dicts, categories):
        stacked = []
        for counter in counter_dicts:
            stacked.append([counter.get(cat, 0) for cat in categories])
        return np.array(stacked)

    dn_categories = ["daytime", "nighttime", "twilight"]
    season_categories = ["winter", "spring", "summer", "autumn"]

    dn_stacked = prepare_stacked([lin_dn, hko_dn, lau_dn], dn_categories)
    season_stacked = prepare_stacked([lin_season, hko_season, lau_season], season_categories)

    sites = ["LIN", "HKO", "LAU"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    bottom = np.zeros(len(sites))

    colors_dn = ["#1f77b4", "#ff7f0e", "#9467bd"]  # day, night, twilight

    for i, cat in enumerate(dn_categories):
        ax.bar(sites, dn_stacked[:, i], bottom=bottom, label=cat, color=colors_dn[i])
        bottom += dn_stacked[:, i]

    ax.set_ylabel("Number of launches")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=2)
    
    ax = axes[1]
    bottom = np.zeros(len(sites))

    colors_season = ["#2ca02c", "#d62728", "#8c564b", "#17becf"]

    for i, cat in enumerate(season_categories):
        ax.bar(sites, season_stacked[:, i], bottom=bottom, label=cat, color=colors_season[i])
        bottom += season_stacked[:, i]

    ax.set_ylabel("Number of launches")
    ax.set_ylabel('')        # remove axis label
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.775,
        bottom=0.10,
        left=0.10,
        right=0.95,
        hspace=0.20,
        wspace=0.25   # leggermente più largo del tuo 0.223
        )
    plt.show()

    print("\n================ FINAL QC SUMMARY ================\n")

    def count_nonempty(df_list):
        return sum([not df.empty for df in df_list])

    # --- Final counts per site ---
    def final_counts(site_dict, site_name):
        total = len(site_dict)

        # Missing data
        md = [gdp.qc_results["missing_data"] for pid, gdp in site_dict.items()]
        missing_profiles = count_nonempty(md)

        # Altitude drops
        ad = [gdp.qc_results["altitude_drops"] for pid, gdp in site_dict.items()]
        alt_drop_profiles = count_nonempty(ad)

        # Outliers
        out = [gdp.qc_results["detect_outliers"] for pid, gdp in site_dict.items()]
        outlier_profiles = count_nonempty(out)

        # Physics constraints
        pc = [gdp.qc_results["physics_constraint"] for pid, gdp in site_dict.items()]
        physics_profiles = count_nonempty(pc)

        print(f"{site_name}:")
        print(f"  Total profiles after filtering: {total}")
        print(f"  Profiles with missing data:     {missing_profiles}")
        print(f"  Profiles with altitude drops:   {alt_drop_profiles}")
        print(f"  Profiles with outliers:         {outlier_profiles}")
        print(f"  Profiles with physics issues:   {physics_profiles}")
        print()

    # Print summary for each site
    final_counts(hko, "HKO")
    final_counts(lau, "LAU")
    final_counts(lin, "LIN")

    print("==================================================\n")
