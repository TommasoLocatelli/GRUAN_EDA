import numpy as np
import pandas as pd
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import gruanpy as gp
from applications.pblh_unc.methodology import fit_ssm
from tqdm import tqdm

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
pkl_path = r"applications/pblh_unc/pkls/gdp_2024_POT-RS-02_2024.pkl"

with open(pkl_path, "rb") as f:
    dataset = pickle.load(f)

print("Dataset Loaded:", len(dataset))

# ---------------------------------------------------------
# Optimizers to test
# ---------------------------------------------------------
methods = ["lbfgs", "powell"]

# ---------------------------------------------------------
# Test each optimizer and rank them
# ---------------------------------------------------------
results_dict = {}

for pid, gdp in tqdm(dataset.items()):

    print(f"\nProcessing profile: {pid}")

    method_results = []

    for method in methods:
        print(f"\nTesting optimizer: {method}")

        try:
            model, results = fit_ssm(gdp, method=method, iterations=100)
            ret = results.mle_retvals

            converged  = ret.get("converged", False)
            warnflag   = ret.get("warnflag", None)
            iterations = ret.get("iterations", None)
            llf        = getattr(results, "llf", np.nan)

            method_results.append({
                "method": method,
                "converged": converged,
                "warnflag": warnflag,
                "iterations": iterations,
                "llf": llf
            })

            print(f"  converged:  {converged}")
            print(f"  warnflag:   {warnflag}")
            print(f"  iterations: {iterations}")
            print(f"  llf:        {llf}")

        except Exception as e:
            print(f"  ERROR in {method}: {e}")
            method_results.append({
                "method": method,
                "converged": False,
                "warnflag": 99,
                "iterations": None,
                "llf": np.nan
            })

    # ---------------------------------------------------------
    # Rank optimizers
    # ---------------------------------------------------------
    df = pd.DataFrame(method_results)

    df_ranked = df.sort_values(
        by=["converged", "llf", "warnflag", "iterations"],
        ascending=[False, False, True, True]
    )

    print("\nRanking:")
    print(df_ranked)

    results_dict[pid] = df_ranked

# ---------------------------------------------------------
# Overall ranking across all profiles
# ---------------------------------------------------------

all_rows = []

for pid, df in results_dict.items():
    for _, row in df.iterrows():
        all_rows.append({
            "pid": pid,
            "method": row["method"],
            "converged": row["converged"],
            "warnflag": row["warnflag"],
            "iterations": row["iterations"],
            "llf": row["llf"]
        })

overall_df = pd.DataFrame(all_rows)

# Aggregate statistics per optimizer
summary = overall_df.groupby("method").agg(
    total_profiles=("pid", "count"),
    total_converged=("converged", "sum"),
    convergence_rate=("converged", "mean"),
    mean_llf=("llf", "mean"),
    mean_warnflag=("warnflag", "mean"),
    mean_iterations=("iterations", "mean")
).reset_index()

# Rank optimizers globally
overall_ranked = summary.sort_values(
    by=["total_converged", "mean_llf", "mean_warnflag", "mean_iterations"],
    ascending=[False, False, True, True]
)

print("\n==============================")
print("OVERALL OPTIMIZER RANKING")
print("==============================")
print(overall_ranked)
