import sys
import os
import time
import random
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import gruanpy as gp

# ---------------------------------------------------------
# Specify the pickle file you want to load
# ---------------------------------------------------------
pkl_path = r"papers\pblh_unc\gdp_2024_LIN-RS-01_2024.pkl"

# ---------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------
if not os.path.isfile(pkl_path):
    raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

with open(pkl_path, "rb") as f:
    dataset = pickle.load(f)

# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------
print(f"Loaded dataset from: {pkl_path}")
print(f"Number of profiles: {len(dataset)}")

# Optional: list the Product.Id keys
print("\nProduct.Id entries:")
for pid, gdp in dataset.items():
    print(" -", pid)
    print(gdp.data.head())
