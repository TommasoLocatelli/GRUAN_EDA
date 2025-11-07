"""
Example code to perform temporal gridding on spatially gridded GDP data and plot temperature trends over time for each altitude.
Some bugs still need to be fixed.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Need to download and unzip a folder with RS41 GDPs from GRUAN website
gdp_folder=r'gdp\products_RS41-GDP-1_LIN_2017' # Path to the folder
gdp_files = [os.path.join(gdp_folder, f) for f in os.listdir(gdp_folder) if f.endswith('.nc')]
gdps=[]
for file in tqdm(gdp_files[:10] , desc="Reading GDPs"):
    gdps.append(gp.read(file))

# PARAMETERS
TARGET_COLUMNS = ['temp']
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
