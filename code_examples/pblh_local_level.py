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
coef = 1000
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

pblhs=gp.pblh_values(gp.apply_pblh_methods(data))
smoothed_pblhs=gp.pblh_values(gp.apply_pblh_methods(smoothed_data))
sim_pblhs=[gp.pblh_values(gp.apply_pblh_methods(sim)) for sim in simulations]
pm_sim_pblh = np.nanmean([sim_pblh[0] for sim_pblh in sim_pblhs], axis=0)
theta_sim_pblh = np.nanmean([sim_pblh[1] for sim_pblh in sim_pblhs], axis=0)
rh_sim_pblh = np.nanmean([sim_pblh[2] for sim_pblh in sim_pblhs], axis=0)
q_sim_pblh = np.nanmean([sim_pblh[3] for sim_pblh in sim_pblhs], axis=0)
Ri_sim_pblh = np.nanmean([sim_pblh[4] for sim_pblh in sim_pblhs], axis=0)
pm_var=np.nanvar([sim_pblh[0] for sim_pblh in sim_pblhs], axis=0)
theta_var=np.nanvar([sim_pblh[1] for sim_pblh in sim_pblhs], axis=0)
rh_var=np.nanvar([sim_pblh[2] for sim_pblh in sim_pblhs], axis=0)
q_var=np.nanvar([sim_pblh[3] for sim_pblh in sim_pblhs], axis=0)
Ri_var=np.nanvar([sim_pblh[4] for sim_pblh in sim_pblhs], axis=0)
pm_unc = np.sqrt(pm_var)*2
theta_unc = np.sqrt(theta_var)*2
rh_unc = np.sqrt(rh_var)*2
q_unc = np.sqrt(q_var)*2
Ri_unc = np.sqrt(Ri_var)*2
pblh_unc = (pm_unc, theta_unc, rh_unc, q_unc, Ri_unc)
avg_sim_pblh = (pm_sim_pblh, theta_sim_pblh, rh_sim_pblh, q_sim_pblh, Ri_sim_pblh)

print(pblhs)
print(smoothed_pblhs)
print(avg_sim_pblh)
print(pblh_unc)

plt.figure(figsize=(10, 12))
plt.suptitle(f'RS41-GDP: {where}, {when}'#, {file_index}'
            , fontsize=20)

# Scatter plot for temperature vs altitude
plt.scatter(data['temp'], data['alt'], label='Original', alpha=0.5, s=1)
plt.scatter(smoothed_data['temp'], smoothed_data['alt'], label='Smoothed', alpha=0.5, s=1)
for sim in simulations:
    plt.plot(sim['temp'], sim['alt'], alpha=0.2, color='gray')
plt.legend()
plt.ylabel('Altitude (m)')
plt.xlabel('Temperature (K)')
plt.title('Temperature vs Altitude')
plt.tight_layout()
plt.show()