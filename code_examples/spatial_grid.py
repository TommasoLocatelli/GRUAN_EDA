import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import GRUANpy as Gp
import pandas as pd
import matplotlib.pyplot as plt

gp=Gp()
file_path=r'gdp\POT-RS-02_2_RS41-GDP_001_20250303T230900_1-000-001.nc'
gdp=gp.read(file_path)

bold_columns=['alt', 'alt_uc',
            #'lat', 'lat_uc', 'lon', 'lon_uc',
            'press', 'press_uc',
            'temp', 'temp_uc',
            'rh', 'rh_uc',
            #'wdir', 'wdir_uc', 'wspeed', 'wspeed_uc'
            ]

bin_column = 'press'
target_columns = ['temp', 'rh']
bin_size = (gdp.data[bin_column].max()-gdp.data[bin_column].min()) / 100


ggd = gp.spatial_gridding(gdp, bin_column, target_columns, bin_size)

print(ggd.metadata.head(50))
print(ggd.data.head())
"""
for column in target_columns:
    plt.figure()
    plt.plot(binned_data[column], binned_data['bin'], marker='o', label=column)
    plt.xlabel(column)
    plt.ylabel(bin_column)
    plt.title(f'{column} vs {bin_column}')
    plt.legend()
    plt.grid()
    plt.show()
"""