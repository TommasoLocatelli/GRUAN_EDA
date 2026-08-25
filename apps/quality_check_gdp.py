import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gruanpy as gp
from apps.visual_config.color_map import map_labels_to_colors

# -------------------------
# MAIN SCRIPT
# -------------------------

folder = r"data\products_RS41-GDP-1_POT_2025"
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")
]

for file_path in file_paths[1:2]:
    print(f"\nReading: {file_path}")
    gdp = gp.read_gdp(file_path)

    # Apply upper bounds
    gdp.data = gp.upper_bound(gdp.data)
    print(gdp.qc_results)