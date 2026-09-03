import pickle
from pprint import pprint

# ---------------------------------------------------------
# Load the pickle file
# ---------------------------------------------------------

pkl_path = "ssm_fit_lll_llt_hko_lau_lin_2024.pkl"

with open(pkl_path, "rb") as f:
    results = pickle.load(f)

print("\nLoaded results.")
print("Number of PIDs:", len(results))
print("PID list:", list(results.keys()))
print("\n")


# ---------------------------------------------------------
# Inspect one PID
# ---------------------------------------------------------

# Choose first PID
pid = list(results.keys())[0]
pid_results = results[pid]

print(f"Inspecting PID: {pid}")
print("Variables available:", list(pid_results.keys()))
print("\n")


# ---------------------------------------------------------
# Inspect one variable
# ---------------------------------------------------------

var = list(pid_results.keys())[0]
var_results = pid_results[var]

print(f"Inspecting variable: {var}")
print("Models available:", list(var_results.keys()))
print("\n")


# ---------------------------------------------------------
# Inspect one model (example: LLT MLE)
# ---------------------------------------------------------

model_name = "llt_mle"
model, fit_result = var_results[model_name]

print(f"Model: {model_name}")
print("Parameter estimates:")
pprint(fit_result.params)

print("\nLog-likelihood:", fit_result.llf)
print("Converged:", fit_result.mle_retvals.get("converged", None))
print("Iterations:", fit_result.mle_retvals.get("iterations", None))
print("\n")


# ---------------------------------------------------------
# Extract smoothed states
# ---------------------------------------------------------

if hasattr(fit_result, "smoother_results"):
    smoothed = fit_result.smoother_results.smoothed_state
    print("Smoothed state shape:", smoothed.shape)

    # Level component
    level = smoothed[0]
    print("Level (first 10 values):", level[:10])

    # Trend component (LLT only)
    if smoothed.shape[0] > 1:
        trend = smoothed[1]
        print("Trend (first 10 values):", trend[:10])

else:
    print("No smoother results available for this model.")
