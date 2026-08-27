import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gruanpy as gp
from gruanpy.plots.color_map import map_labels_to_colors

# -------------------------
# MAIN SCRIPT
# -------------------------

folder = r"data\products_RS41-GDP-1_POT_2025"
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")
]

for file_path in file_paths[0:10]:
    print(f"\nReading: {file_path}")
    gdp = gp.read_gdp(file_path, upper_bound=True,
        columns=gp.COLUMNS_OF_INTEREST)
    for key, value in gdp.qc_results.items():
        print(key, value)

    outliers = gdp.qc_results['detect_outliers']

    for _, row in outliers.iterrows():
        var = row['variable']
        idx = row['indices']          # list of outlier indices

        plt.figure(figsize=(10, 4))

        # full series
        plt.plot(gdp.data[var], label="full series", color="black")

        # outlier points in red
        plt.scatter(idx, gdp.data.loc[idx, var],
                    color="red", s=40, label="outliers", zorder=3)

        plt.title(f"Outliers in {var}")
        plt.xlabel("Index")
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()
        plt.show()