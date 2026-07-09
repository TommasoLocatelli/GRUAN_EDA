import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Folder containing the *_pblh_results.pkl files
RESULTS_FOLDER = r"papers\pblh_unc"

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
# LOAD ALL RESULTS INTO ONE BIG DATAFRAME
# ---------------------------------------------------------

rows = []

for file in result_files:
    with open(file, "rb") as f:
        results = pickle.load(f)

    for pid, info in results.items():

        pblh_info = info["pblh_info"]

        for method in ["pm", "thv", "rh", "ri"]:
            rows.append({
                "pid": pid,
                "site": info["where"],
                "date": info["when"],
                "tod": info["tod"],
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
print(df.head())

# ---------------------------------------------------------
# BASIC ANALYSIS
# ---------------------------------------------------------

# 1. Summary statistics per site and method
summary = df.groupby(["site", "method"]).agg({
    "value": ["mean", "std"],
    "median": ["mean", "std"],
    "unc_width": ["mean", "std"]
}).round(2)

print("\n=== SUMMARY PER SITE & METHOD ===")
print(summary)

# 2. Day vs Night comparison
tod_summary = df.groupby(["site", "method", "tod"]).agg({
    "value": "mean",
    "median": "mean",
    "unc_width": "mean"
}).round(2)

print("\n=== DAY vs NIGHT ===")
print(tod_summary)

# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

sns.set(style="whitegrid")

# 3. Distribution of PBLH per method
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="method", y="median")
plt.title("Distribution of PBLH (median MC) per method")
plt.ylabel("PBLH [m]")
plt.show()

# 4. Uncertainty width per method
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="method", y="unc_width")
plt.title("Uncertainty width per method")
plt.ylabel("High - Low [m]")
plt.show()

# 5. Site comparison
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="site", y="median", hue="method")
plt.title("PBLH by site and method")
plt.xticks(rotation=45)
plt.show()

# 6. Day vs Night
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="tod", y="median", hue="method")
plt.title("Day vs Night PBLH")
plt.show()

# ---------------------------------------------------------
# OPTIONAL: Flatten MC samples into long format
# ---------------------------------------------------------

long_rows = []

for _, row in df.iterrows():
    samples = row["samples"]
    for s in samples:
        long_rows.append({
            "pid": row["pid"],
            "site": row["site"],
            "date": row["date"],
            "tod": row["tod"],
            "method": row["method"],
            "sample": s
        })

df_samples = pd.DataFrame(long_rows)

print("\nMC sample dataframe shape:", df_samples.shape)
print(df_samples.head())

# Example: KDE plot of MC samples
plt.figure(figsize=(10,6))
sns.kdeplot(data=df_samples, x="sample", hue="method", fill=True)
plt.title("MC sample distributions per method")
plt.show()
