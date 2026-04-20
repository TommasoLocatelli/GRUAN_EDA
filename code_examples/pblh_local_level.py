import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
import pandas as pd
from code_examples.visual_config.color_map import map_labels_to_colors
from matplotlib.patches import Ellipse
import statsmodels.api as sm
from ssm.local_level import MeasurementLocalLevel

paths = [
    r'gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc',
    r'gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc'
]
gdp=gp.read(paths[0])
upper_bound=gp._find_upper_bound(gdp.data[['alt']], upper_bound=3000, return_value=True) # find the PBLH upper bound for profile
data = gdp.data[gdp.data['alt'] <= upper_bound]  # Limit to first 3.5 km
where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0] # location
when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0] # time
when=when[0:10]+' '+when[11:19]

start = data['time'].values[0]
time = data['time'].values
seconds = (time - start) / np.timedelta64(1, 's')
seconds = seconds.astype(float)

measurements_names = ['alt', 'temp', 'rh', 'press', 'wspeed', 'wdir']
measurements = np.column_stack([data[name].values for name in measurements_names])
measurements_unc =np.column_stack([data[name+'_uc'].values for name in measurements_names])
# Extract measurement variances
measurement_var = (measurements_unc * 0.5)**2
# tune the measurement variance to see more effect of the model
coef = 1
if coef != 1:
    measurement_var *= coef
    measurements_unc = (measurement_var**0.5)*2

mod=MeasurementLocalLevel(measurements, measurement_var, measurements_names)
# Fit it using MLE with a fixed sequence of measurement variances
res = mod.fit(method='lbfgs',
            maxiter=50,
            full_output=1,
            disp=5)
print(res.summary())

smoothed_data = pd.DataFrame(res.smoothed_state.T, columns=measurements_names)
smoothed_data['time'] = time
smoothed_data['seconds'] = seconds

sim = mod.simulation_smoother() # default method is KFS; (method='cfa')  # can specify CFA method
nsimulations = 10
simulations = []
for i in range(nsimulations):
    sim.simulate()
    sim_data = pd.DataFrame(sim.simulated_state.T, columns=measurements_names)
    sim_data['time'] = time
    sim_data['seconds'] = seconds
    simulations.append(sim_data)

print(data.head())
print(smoothed_data.head())
print(sim_data.head())