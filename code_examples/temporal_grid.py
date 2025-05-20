import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from gruanpy.data_models.ggd import Ggd

# Need to download and unzip a folder with RS41 GDPs from GRUAN website
gdp_folder=r'gdp\products_RS41-GDP-1_LIN_2017' # Path to the folder
gdp_files = [os.path.join(gdp_folder, f) for f in os.listdir(gdp_folder) if f.endswith('.nc')]
gdps=[]
for file in tqdm(gdp_files[:100] , desc="Reading GDPs"):
    gdps.append(gp.read(file))
print(gdps[0].data.head())

# PARAMETERS
TARGET_COLUMNS = ['temp']#, 'rh']
BIN_COLUMN = 'alt' # spatial bin in spatial gridding
MANDATORY_LEVELS_FLAG = True
LVL='mand_lvl' # spatial bin in temporal gridding
if not MANDATORY_LEVELS_FLAG:
    LVL = BIN_COLUMN+'_bin' # spatial bin in temporal gridding

# Variable Spatial Gridding
ggds=[]
for gdp in tqdm(gdps, desc="Spatial Gridding"):
    ggd = gp.spatial_gridding(gdp, BIN_COLUMN, TARGET_COLUMNS, bin_size=100, mandatory_levels_flag=MANDATORY_LEVELS_FLAG)
    ggds.append(ggd)
print(ggds[0].data.head())

# Temporal Gridding
tggd=gp.temporal_gridding(ggds, TARGET_COLUMNS, bin_size=7, lvl_column=LVL)
print(tggd.data.head())

# Plot temperature trend over time for each altitude
unique_alts = tggd.data[LVL].unique()
for alt in unique_alts[:10:]:
    alt_data = tggd.data[tggd.data[LVL] == alt]
    plt.plot(alt_data['time'], alt_data['temp'], label=alt)
    plt.fill_between(alt_data['time'], alt_data['temp'] - alt_data['temp_uc'], alt_data['temp'] + alt_data['temp_uc'], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Temperature (K)')
plt.legend()
plt.title('Temperature Trend over Time for Each Altitude')
plt.show()

"""
output_folder = 'outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_folder, f'temperature_trend_over_time_{timestamp}.png')
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
"""
