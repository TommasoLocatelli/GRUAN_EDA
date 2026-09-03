import dill
from collections import defaultdict

# ---------------------------------------------------------
# Load the pickle file with dill
# ---------------------------------------------------------

pkl_path = "ssm_fit_lll_llt_HKO_2024.pkl"

with open(pkl_path, "rb") as f:
    results = dill.load(f)

print("Loaded results.")
print("Number of PIDs:", len(results))
print("\n")

# ---------------------------------------------------------
# Count convergence across all PIDs / variables / models
# ---------------------------------------------------------

convergence_summary = defaultdict(int)
not_converged_per_var = defaultdict(int)

for pid, pid_results in results.items():
    for var, var_results in pid_results.items():
        for model_name, (model, fit_result) in var_results.items():

            converged = fit_result.mle_retvals.get("converged", None)

            if converged is True:
                convergence_summary["converged"] += 1
            elif converged is False:
                convergence_summary["not_converged"] += 1
                not_converged_per_var[var] += 1
            else:
                convergence_summary["unknown"] += 1

# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("Convergence summary:")
print(f"  Converged:      {convergence_summary['converged']}")
print(f"  Not converged:  {convergence_summary['not_converged']}")
print(f"  Unknown flag:   {convergence_summary['unknown']}")
print("\n")

# ---------------------------------------------------------
# Print not-converged counts per variable
# ---------------------------------------------------------

print("Not converged per variable:")
for var, count in not_converged_per_var.items():
    print(f"  {var}: {count}")

