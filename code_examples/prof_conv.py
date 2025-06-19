import sys
import os
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

file_path=r'gdp\data_examples\LIN-RS-01_2_RS41-GDP_001_20141209T120000_1-009-002.nc'
file_path=r'gdp\products_RS41-GDP-1_LIN_2017\LIN-RS-01_2_RS41-GDP_001_20170103T000000_1-003-001.nc'
#file_path=r'gdp\products_RS41-GDP-1_LIN_2017\LIN-RS-01_2_RS41-GDP_001_20170106T120000_1-003-002.nc'
gdp=gp.read(file_path)

data = gdp.data[['alt', 'temp']].sort_values(by='alt').reset_index(drop=True)
kernel_size = 11
uniform_kernel = np.array([1/kernel_size for _ in range(kernel_size)])
increasing_kernel = np.array([(i+1)/kernel_size for i in range(-kernel_size//2, kernel_size//2)])
bimodal_kernel = np.array([-1 if i < kernel_size//2 else
                           0 if i == kernel_size//2 else 1
                           for i in range(kernel_size)])
border_delta_kernel = np.array([-1/kernel_size if i==0 else
                                0 if i < kernel_size-1 else
                                1/kernel_size for i in range(kernel_size)])
kernel=border_delta_kernel #how to compute grided lapse rate directly from a convolution kernel?
print(kernel)

temp_prof = data['temp'].to_numpy()
smoothed_temp = np.convolve(temp_prof, kernel, mode='valid')
smoothed_temp = np.concatenate((np.full(kernel_size//2, np.nan), smoothed_temp, np.full(kernel_size//2, np.nan)))
print(len(temp_prof), len(kernel), sum(kernel), len(smoothed_temp))
data['smoothed_temp'] = smoothed_temp

plt.figure(figsize=(8, 6))
plt.plot(data['temp'], data['alt'], label='Original Temp', alpha=0.7)
plt.plot(data['smoothed_temp'], data['alt'], label='Smoothed Temp', linewidth=2)
plt.ylabel('Altitude (m)')
plt.xlabel('Temperature (°C)')
plt.title('Temperature and Smoothed Temperature vs Altitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
