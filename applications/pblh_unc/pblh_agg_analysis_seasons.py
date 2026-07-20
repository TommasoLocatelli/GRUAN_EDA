import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Folder containing the *_pblh_results.pkl files
RESULTS_FOLDER = r"applications\pblh_unc\pkls"

# Collect all result pickles
result_files = [
    os.path.join(RESULTS_FOLDER, f)
    for f in os.listdir(RESULTS_FOLDER)
    if f.endswith("_pblh_results.pkl")
]

print("Found result files:")
for f in result_files:
    print(" -", f)

# ---------------------------------------------------------
# FUNCTION: MAP DATE TO SEASON
# ---------------------------------------------------------

def get_season(date_str):
    """Convert date string to season."""
    try:
        dt = pd.to_datetime(date_str)
    except:
        return None

    m = dt.month
    if m in [12, 1, 2]:
        return "winter"
    elif m in [3, 4, 5]:
        return "spring"
    elif m in [6, 7, 8]:
        return "summer"
    elif m in [9, 10, 11]:
        return "autumn"
    return None

# ---------------------------------------------------------
# LOAD ALL RESULTS INTO ONE BIG DATAFRAME
# ---------------------------------------------------------

rows = []

for file in result_files:
    with open(file, "rb") as f:
        results = pickle.load(f)

    for pid, info in results.items():

        pblh_info = info["pblh_info"]
        season = get_season(info["when"])

        for method in ["pm", "thv", "rh", "ri"]:
            rows.append({
                "pid": pid,
                "site": info["where"],
                "date": info["when"],
                "season": season,
                "method": method,

                # deterministic GRUAN value
                "value": pblh_info[method]["value"],

                # MC statistics
                "median": pblh_info[method]["median"],
                "low": pblh_info[method]["low"],
                "high": pblh_info[method]["high"],
                "samples": pblh_info[method]["samples"],

                # uncertainty width
                "unc_width": pblh_info[method]["high"] - pblh_info[method]["low"],
            })

df = pd.DataFrame(rows)

print("\nLoaded profiles:", df["pid"].nunique())
print("Total rows:", len(df))
print(df.info())

# ---------------------------------------------------------
# BASIC ANALYSIS
# ---------------------------------------------------------

sns.set(style="whitegrid")

# Skip Potenza
sites = [s for s in sorted(df["site"].unique()) if s != "Potenza"]

methods = ["pm", "thv", "rh", "ri"]
season_order = ["winter", "spring", "summer", "autumn"]

season_palette = {
    "winter": "#1f77b4",
    "spring": "#2ca02c",
    "summer": "#ff7f0e",
    "autumn": "#8c564b",
}

# --- GLOBAL Y LIMITS FOR ALL ROWS ---
ymin = min(df["value"].min(), df["median"].min(), df["unc_width"].min())
ymax = max(df["value"].max(), df["median"].max(), df["unc_width"].max())

# ---------------------------------------------------------
# FUNCTION TO COMPUTE FULL STAT TABLE (value, median, unc_width)
# ---------------------------------------------------------

def compute_full_stats(df_site):
    df_clean = df_site.dropna(subset=["value", "median", "unc_width"], how="any")

    rows = []

    for method in methods:
        for season in season_order:
            subset = df_clean[(df_clean["method"] == method) &
                              (df_clean["season"] == season)]

            if len(subset) == 0:
                continue

            def med_IQR(series):
                s = series.dropna()
                med = np.median(s)
                q25 = np.percentile(s, 25)
                q75 = np.percentile(s, 75)
                return med, q75 - q25

            val_med, val_IQR = med_IQR(subset["value"])
            mcm_med, mcm_IQR = med_IQR(subset["median"])
            unc_med, unc_IQR = med_IQR(subset["unc_width"])

            rows.append({
                "method": method,
                "season": season,
                "value_median": val_med,
                "value_IQR": val_IQR,
                "mcm_median": mcm_med,
                "mcm_IQR": mcm_IQR,
                "unc_median": unc_med,
                "unc_IQR": unc_IQR
            })

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# LOOP OVER SITES
# ---------------------------------------------------------

for site in sites:
    df_site = df[df["site"] == site]

    # ---- Compute and save full stats table ----
    stats_table = compute_full_stats(df_site)
    stats_table.to_csv(f"{site}_SEASON_stats.csv", index=False)
    print(f"Saved table: {site}_SEASON_stats.csv")

    # ---- Create figure ----
    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(18, 14),
        sharey=True
    )

    # ---------------------------------------------------------
    # ROW 1: deterministic values
    # ---------------------------------------------------------
    for ax, method in zip(axes[0], methods):
        df_m = df_site[df_site["method"] == method]

        sns.violinplot(
            data=df_m,
            x="season",
            y="value",
            order=season_order,
            palette=season_palette,
            cut=0,
            inner="quartile",
            ax=ax
        )

        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("")
        ax.set_ylabel("Standard PBLH [m]")
        ax.set_title(f"{method.upper()}")
        ax.tick_params(axis="x", rotation=25)

    # ---------------------------------------------------------
    # ROW 2: median MC estimates
    # ---------------------------------------------------------
    for ax, method in zip(axes[1], methods):
        df_m = df_site[df_site["method"] == method]

        sns.violinplot(
            data=df_m,
            x="season",
            y="median",
            order=season_order,
            palette=season_palette,
            cut=0,
            inner="quartile",
            ax=ax
        )

        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("")
        ax.set_ylabel("MCM PBLH [m]")
        ax.set_title("")
        ax.tick_params(axis="x", rotation=25)

    # ---------------------------------------------------------
    # ROW 3: uncertainty width
    # ---------------------------------------------------------
    for ax, method in zip(axes[2], methods):
        df_m = df_site[df_site["method"] == method]

        sns.violinplot(
            data=df_m,
            x="season",
            y="unc_width",
            order=season_order,
            palette=season_palette,
            cut=0,
            inner="quartile",
            ax=ax
        )

        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("")
        ax.set_ylabel("PBLH Unc. Width [m]")
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("")

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    fig.suptitle(f"PBLH Distribution by Season - {site}", fontsize=16)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    # ---- Save figure ----
    fig.savefig(f"{site}_PBLH_SEASON_violin.png", dpi=300)
    print(f"Saved figure: {site}_PBLH_SEASON_violin.png")

    plt.close(fig)
