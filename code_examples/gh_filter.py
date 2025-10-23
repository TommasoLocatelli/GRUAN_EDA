
# ISPIRED BY https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

import sys
import os
import matplotlib.pyplot as plt
import numpy.random as random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp

folder = r'gdp\products_RS41-GDP-1_POT_2025'
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:5]:
    gdp = gp.read(file_path)
    gdp.data = gdp.data[gdp.data['alt'] <= 2000]  # Limit to first 10 km for speed
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
    gdp.data['temp_noisy'] = gdp.data['temp']+random.normal(0, 0.5, size=gdp.data['temp'].shape)

    initial_guess = gdp.data['temp_noisy'][0]+1 # K
    initial_alt = gdp.data['alt'][0] # m
    previous_alt = initial_alt # m
    gain_rate = 0.01 # K/m
    temp_scale = 0.4 
    gain_scale = 0.6
    estimates = []
    estimates.append(initial_guess)

    for alt, temp in zip(gdp.data['alt'][1:], gdp.data['temp_noisy'][1:]): 
        #prediction step
        alt_step = alt - previous_alt
        temp_estimate = initial_guess + gain_rate * alt_step

        #update step
        residual = temp - temp_estimate
        gain_rate += gain_scale * residual / alt_step
        temp_estimate += temp_scale * residual
        estimates.append(temp_estimate)

    #plot tempo over altitude
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(gdp.data['temp_noisy'], gdp.data['alt'], label='Temperature Profile', linestyle=':')
    ax.plot(gdp.data['temp'], gdp.data['alt'], label='True Temperature')
    ax.plot(estimates, gdp.data['alt'], label='Filter Estimate', linestyle='--')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Temperature Profile\nSite: {where}, Time: {when}')
    ax.grid()
    plt.legend()
    plt.show()
    #break
