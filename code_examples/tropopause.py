import sys
import os
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None

file_path=r'gdp\data_examples\LIN-RS-01_2_RS41-GDP_001_20141209T120000_1-009-002.nc' 
file_path=r'gdp\products_RS41-GDP-1_LIN_2017\LIN-RS-01_2_RS41-GDP_001_20170103T000000_1-003-001.nc' 
file_path=r'gdp\products_RS41-GDP-1_LIN_2017\LIN-RS-01_2_RS41-GDP_001_20170106T120000_1-003-002.nc'
gdp=gp.read(file_path)

# temperature vs altitude
plt.figure(figsize=(10, 6))

sc = plt.scatter(gdp.data['temp'], gdp.data['alt'], c=gdp.data['press'], cmap='coolwarm', marker='o')
plt.colorbar(sc, label='Pressure '+gdp.variables_attrs[gdp.variables_attrs['variable']=='press']['units'].values[0])

plt.xlabel('Temperature '+gdp.variables_attrs[gdp.variables_attrs['variable']=='temp']['units'].values[0])
plt.ylabel('Altitude m')
plt.title('Altitude vs. Temperature with Pressure as Color')
plt.grid(True)
plt.show()

"""
Tropopause Calculation Exercise

https://www.ncl.ucar.edu/Document/Functions/Built-in/trop_wmo.shtml

"""

standard_name = {
        'alt': 'geopotential_height',
        'temp': 'air_temperature'
    }

units = gdp.variables_attrs[gdp.variables_attrs['standard_name'].isin(standard_name.values())][['units','long_name', 'standard_name']]
    
def gdp_trop_wmo(gdp, lapse_rate=2.0):
    """
    Calculate the tropopause using WMO criteria.
    """
    # Check if data is sorted by altitude
    if not gdp.data['alt'].is_monotonic_increasing:
        print("Warning: Data is not sorted by altitude.")

    data = gdp.data[['alt', 'temp']]#.sort_values(by='alt').reset_index(drop=True)
    data['alt_lag'] = data['alt'].shift(1)
    data['temp_lag'] = data['temp'].shift(1)
    data['delta_alt'] = data['alt'].diff().fillna(1)
    data['delta_temp'] = data['temp'].diff().fillna(0)
    data['lapse_rate'] = - 1000 * data['delta_temp'] / data['delta_alt']
    data = data.iloc[1:].reset_index(drop=True)

    trop_candidate = data[data['lapse_rate'] < lapse_rate]
    trop_height = None
    for index, row in trop_candidate.iterrows():
        window = data[(data['alt'] >= row['alt']) & (data['alt'] <= row['alt'] + 2000)]
        if window['lapse_rate'].max() < lapse_rate and window['alt'].max() - window['alt'].min() > 1999:
            print(f"Tropopause found at {row['alt']} m")
            trop_height = row['alt']
            break
    
    return trop_height, data

trop_height, data = gdp_trop_wmo(gdp)

plt.figure(figsize=(6, 4))
plt.axhline(2, color='red', linestyle='--', label='Lapse Rate = 2 K/km')
if trop_height:
    plt.axvline(trop_height, color='green', linestyle='--', label=f'Tropopause at {trop_height} m')
plt.plot(data['alt'], data['lapse_rate'], label='Lapse Rate (K/m)')
plt.xlabel('Geopotential Height (m)')
plt.ylabel('Lapse Rate (K/m)')
plt.title('Lapse Rate vs Geopotential Height')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(data[abs(data['lapse_rate']) > 50].head(10))