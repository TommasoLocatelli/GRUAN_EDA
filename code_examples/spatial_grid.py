import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import GRUANpy as Gp
import pandas as pd
import matplotlib.pyplot as plt

gp=Gp()
# Need to download one GDP
file_path=r'gdp\POT-RS-02_2_RS41-GDP_001_20250303T230900_1-000-001.nc' # Path to the file
gdp=gp.read(file_path)

main_columns=['alt', 'alt_uc',
            #'lat', 'lat_uc', 'lon', 'lon_uc',
            'press', 'press_uc', 'press_uc_ucor', 'press_uc_scor', 'press_uc_tcor',
            'temp', 'temp_uc', 'temp_uc_ucor', 'temp_uc_scor', 'temp_uc_tcor',
            'rh', 'rh_uc',  'rh_uc_ucor', 'rh_uc_scor', 'rh_uc_tcor'
            #'wdir', 'wdir_uc', 'wspeed', 'wspeed_uc'
            ]

bin_column = 'press' # Choose the binning column (alt or press)
target_columns = ['temp', 'rh']
bin_size = (gdp.data[bin_column].max()-gdp.data[bin_column].min()) / 100

ggd = gp.spatial_gridding(gdp, bin_column, target_columns, bin_size)

print(gdp.data[[col for col in gdp.data.columns if col in main_columns]].head())
print(ggd.data.head())
print(ggd.data.shape)

# Plot original and gridded data
for column in target_columns:
    fig, ax1 = plt.subplots(figsize=(7, 6))
    if bin_column == 'press':
        ax1.set_yscale('log')
        ax1.invert_yaxis()
    ax1.scatter(gdp.data[column], gdp.data[bin_column], label='Original Data', alpha=0.5)
    ax1.scatter(ggd.data[column], ggd.data[bin_column+'_bin'], label='Gridded Data', color='red', alpha=0.5)
    ax1.fill_betweenx(gdp.data[bin_column], gdp.data[column] - gdp.data[column+'_uc'], gdp.data[column] + gdp.data[column+'_uc'], color='blue', alpha=0.2, label='Original Uncertainty')
    ax1.fill_betweenx(ggd.data[bin_column+'_bin'], ggd.data[column] - ggd.data[column+'_uc'], ggd.data[column] + ggd.data[column+'_uc'], color='red', alpha=0.2, label='Gridded Uncertainty')
    ax1.set_xlabel(column.capitalize())
    ax1.set_ylabel(bin_column.capitalize())
    ax1.legend()
    ax1.set_title('Spatial Gridding')

    plt.tight_layout()
    plt.show()
