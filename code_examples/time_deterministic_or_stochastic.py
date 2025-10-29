# this script might determine if atmospheric profiles are sampled by a deterministic or stochastic time interval
# in other words, if the time intervals between consecutive profiles are constant or variable

import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
from visual_config.color_map import map_labels_to_colors
from tqdm import tqdm

folder = r'gdp\products_RS41-GDP-1_POT_2025'
folder = r'gdp\products_RS41-GDP-1_LIN_2017'
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]

files=[]
print(len(file_paths), "files found in folder")
for file_path in file_paths:
    gdp = gp.read(file_path)
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
    time_diffs = gdp.data.sort_values(by='time')['time'].diff().dt.total_seconds().dropna()
    if not time_diffs.nunique() == 1:
        files.append(file_path)
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        # First subplot: Time evolution
        sorted_df = gdp.data.sort_values(by='time').reset_index(drop=True)
        axs[0].plot(sorted_df.index, sorted_df['time'], marker='o')
        axs[0].set_title(f"{where} - {when}\nTime Evolution")
        axs[0].set_ylabel("Time")
        axs[0].grid(True)

        # Second subplot: Time differences
        axs[1].plot(time_diffs, marker='o', color='orange')
        axs[1].set_title("Time Differences Between Profiles")
        axs[1].set_xlabel("Profile Index")
        axs[1].set_ylabel("Time Difference")
        axs[1].grid(True)

        # Adjust layout and show
        plt.tight_layout()
        plt.show()

    #print(len(files))

print("Files with irregular time intervals between profiles:")
print(len(files)/len(file_paths) * 100, "% of total files")