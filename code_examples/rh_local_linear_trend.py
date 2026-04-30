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
from ssm.local_trend import RHLocalLinearTrend

paths = [
    r'gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc',
    r'gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc',
    r'gdp\products_RS41-GDP-1_PAY_2024\PAY-RS-01_2_RS41-GDP_001_20240109T120000_1-002-001.nc',
    r'gdp\products_RS41-GDP-1_TEN_2024\TEN-RS-01_2_RS41-GDP_001_20240103T110000_1-000-001.nc'
]
gdp=gp.read(paths[-1])
upper_bound=gp._find_upper_bound(gdp.data[['alt']], upper_bound=3000, return_value=True) # find the PBLH upper bound for profile
data = gdp.data[gdp.data['alt'] <= upper_bound]  # Limit to first 3.5 km
where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0] # location
when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0] # time
when=when[0:10]+' '+when[11:19]

start = data['time'].values[0]
time = data['time'].values
seconds = (time - start) / np.timedelta64(1, 's')
seconds = seconds.astype(float)

altitude = data['alt'].values
altitude_unc = data['alt_gph_uc_ucor'].values
altitude_var = (altitude_unc * 0.5)**2  # variance of altitude measurements

rh = data['rh'].values
rh_unc = data['rh_uc_ucor'].values
rh_var = (rh_unc * 0.5)**2  # variance of rh measurements

# tune the measurement variance to see more effect of the model
coef = 1
if coef != 1:
    altitude_var *= coef
    altitude_unc = np.sqrt(altitude_var) * 2
    rh_var *= coef
    rh_unc = np.sqrt(rh_var) * 2

# Setup the model
mod = RHLocalLinearTrend(endog=np.column_stack([altitude, rh]), alt_measurement_var=altitude_var, rh_measurement_var=rh_var)

# Fit it using MLE with a fixed sequence of measurement variances
res = mod.fit(method='lbfgs',
            maxiter=50,
            full_output=1,
            disp=5)
print(res.summary())

# Smoothed values
smoothed_altitude = res.smoothed_state[0]
smoothed_alt_vel = res.smoothed_state[1]
smoothed_rh = res.smoothed_state[2]
smoothed_rh_vel = res.smoothed_state[3]
smoother_altitude_unc = np.sqrt(res.smoothed_state_cov[0, 0, :])*2
smoother_alt_vel_unc = np.sqrt(res.smoothed_state_cov[1, 1, :])*2
smoother_rh_unc = np.sqrt(res.smoothed_state_cov[2, 2, :])*2
smoother_rh_vel_unc = np.sqrt(res.smoothed_state_cov[3, 3, :])*2
smoothed_grad_rh = smoothed_rh_vel / smoothed_alt_vel
smoothed_grad_rh_unc = np.sqrt((smoother_rh_vel_unc / smoothed_alt_vel)**2 + (smoothed_rh_vel * smoother_alt_vel_unc / smoothed_alt_vel**2)**2)*2 # Propagated uncertainty

# Compute finite difference velocity and uncertainty
diff_alt = np.diff(altitude)
diff_time = np.diff(seconds)
diff_rh = np.diff(rh)
diff_ratio_alt = diff_alt / diff_time
diff_ratio_rh = diff_rh / diff_time
diff_grad_rh = diff_rh / diff_alt
diff_ratio_alt_unc = np.sqrt(altitude_var[1:] + (altitude_var[:-1]))*2 / diff_time # Propagated uncertainty
diff_ratio_rh_unc = np.sqrt(rh_var[1:] + (rh_var[:-1]))*2 / diff_time # Propagated uncertainty
diff_grad_rh_unc = np.sqrt(rh_var[1:] + (rh_var[:-1]) + (diff_rh/diff_alt)**2 * (altitude_var[1:] + altitude_var[:-1]))*2 / diff_alt # Propagated uncertainty

# Create simulation smoother objects
sim = mod.simulation_smoother() # default method is KFS; (method='cfa')  # can specify CFA method

nsimulations = 100
simulations = []
for i in range(nsimulations):
    sim.simulate()
    sim_alt=sim.simulated_state[0]
    sim_alt_vel=sim.simulated_state[1]
    sim_rh=sim.simulated_state[2]
    sim_rh_vel=sim.simulated_state[3]
    sim_grad_rh = sim_rh_vel / sim_alt_vel
    simulations.append((sim_alt, sim_alt_vel, sim_rh, sim_rh_vel, sim_grad_rh))

pblh_rh = altitude[np.argmin(diff_grad_rh)]
sm_pblh_rh = smoothed_altitude[np.argmin(smoothed_grad_rh)]
simulations_pblh_rh = [sim_alt[np.argmin(sim_grad_rh)] for sim_alt, sim_alt_vel, sim_rh, sim_rh_vel, sim_grad_rh in simulations]
simulations_pblh_rh_avg = np.mean(simulations_pblh_rh)
simulations_pblh_rh_unc = np.std(simulations_pblh_rh)*2
print(f'PBLH (RH Gradient): {pblh_rh:.1f} m')
print(f'PBLH (Smoothed RH Gradient): {sm_pblh_rh:.1f} m')
print(f'PBLH (Simulated RH Gradient): {simulations_pblh_rh_avg:.1f} m ± {simulations_pblh_rh_unc:.1f} m')

# Lots of Plots
 
plt.figure(figsize=(10, 12))
plt.suptitle(f'RS41-GDP: {where}, {when}'#, {file_index}'
            , fontsize=20)

plt.subplot(1, 3, 1)

LOWER_LIMIT=300#180#150
UPPER_LIMIT=400#500#230#310

plt.scatter(rh[LOWER_LIMIT:UPPER_LIMIT], altitude[LOWER_LIMIT:UPPER_LIMIT], label='Observed Vertical Profile', color=map_labels_to_colors['rh'],
            marker='o', s=4)
plt.fill_betweenx(
    altitude[LOWER_LIMIT:UPPER_LIMIT],
    rh[LOWER_LIMIT:UPPER_LIMIT] - rh_unc[LOWER_LIMIT:UPPER_LIMIT],
    rh[LOWER_LIMIT:UPPER_LIMIT] + rh_unc[LOWER_LIMIT:UPPER_LIMIT],
    color=map_labels_to_colors['rh_uc'],
    alpha=0.3,
    label='Measurement uncertainty',
)
plt.plot(smoothed_rh[LOWER_LIMIT:UPPER_LIMIT], smoothed_altitude[LOWER_LIMIT:UPPER_LIMIT], label='Smoothed Vertical Profile', color=map_labels_to_colors['wspeed'],
            #marker='o', s=4
            )
plt.fill_betweenx(
    smoothed_altitude[LOWER_LIMIT:UPPER_LIMIT],
    smoothed_rh[LOWER_LIMIT:UPPER_LIMIT] - smoother_rh_unc[LOWER_LIMIT:UPPER_LIMIT],
    smoothed_rh[LOWER_LIMIT:UPPER_LIMIT] + smoother_rh_unc[LOWER_LIMIT:UPPER_LIMIT],
    color=map_labels_to_colors['wspeed_uc'],
    alpha=0.3,
    label='Smoothed uncertainty',
)
if pblh_rh is not None:
    plt.axhline(pblh_rh, color=map_labels_to_colors['rh'], linestyle='--', 
    label=f'PBLH (Observation): {pblh_rh:.1f} m'
    )
if sm_pblh_rh is not None:
    plt.axhline(sm_pblh_rh, color=map_labels_to_colors['wspeed'], linestyle=':', 
    label=f'PBLH (Smoothed): {sm_pblh_rh:.1f} m'
    )
if simulations_pblh_rh_avg is not None:
    plt.axhline(simulations_pblh_rh_avg, color='gray', linestyle='--', 
    label=f'PBLH (MCM): {simulations_pblh_rh_avg:.1f} m ± {simulations_pblh_rh_unc:.1f} m'
    )
if simulations_pblh_rh_unc is not None:
    plt.fill_betweenx(
        y=[simulations_pblh_rh_avg - simulations_pblh_rh_unc, simulations_pblh_rh_avg + simulations_pblh_rh_unc],
        x1=0,
        x2=100,#120,
        color='gray',
        alpha=0.2,
        label=f'MCM PBLH uncertainty: {simulations_pblh_rh_unc:.1f} m',
    )

#for i in range(nsimulations):
#    plt.plot(simulations[i][2][LOWER_LIMIT:UPPER_LIMIT], simulations[i][0][LOWER_LIMIT:UPPER_LIMIT], color='gray', alpha=0.05, label='Simulated Vertical Profile' if i == 0 else None, zorder=1)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Altitude (m)')
plt.title('RH Vertical Profile with PBLH Estimates')
plt.legend(loc='upper left')
plt.grid(True)
#plt.xlim(18,80)
#plt.xlim(60,110)
plt.xlim(25,55)
plt.subplot(1, 3, 2)

plt.scatter(diff_grad_rh[LOWER_LIMIT:UPPER_LIMIT], altitude[LOWER_LIMIT:UPPER_LIMIT], marker='o', s=4, label='Finite difference RH gradient', color=map_labels_to_colors['rh'])
plt.fill_betweenx(
    altitude[LOWER_LIMIT:UPPER_LIMIT],
    diff_grad_rh[LOWER_LIMIT:UPPER_LIMIT] - diff_grad_rh_unc[LOWER_LIMIT:UPPER_LIMIT],
    diff_grad_rh[LOWER_LIMIT:UPPER_LIMIT] + diff_grad_rh_unc[LOWER_LIMIT:UPPER_LIMIT],
    color=map_labels_to_colors['rh_uc'],
    alpha=0.3,
    label='Finite difference RH gradient uncertainty',
)
plt.plot(smoothed_grad_rh[LOWER_LIMIT:UPPER_LIMIT], smoothed_altitude[LOWER_LIMIT:UPPER_LIMIT], label='Smoothed RH gradient', color=map_labels_to_colors['wspeed'], 
        #marker='o', s=4
        )
plt.fill_betweenx(
    smoothed_altitude[LOWER_LIMIT:UPPER_LIMIT],
    smoothed_grad_rh[LOWER_LIMIT:UPPER_LIMIT] - smoothed_grad_rh_unc[LOWER_LIMIT:UPPER_LIMIT],
    smoothed_grad_rh[LOWER_LIMIT:UPPER_LIMIT] + smoothed_grad_rh_unc[LOWER_LIMIT:UPPER_LIMIT],
    color=map_labels_to_colors['wspeed_uc'],
    alpha=0.3,
    label='Smoothed RH gradient uncertainty',
)
for i in range(nsimulations):
    plt.plot(simulations[i][4][LOWER_LIMIT:UPPER_LIMIT], simulations[i][0][LOWER_LIMIT:UPPER_LIMIT], color='gray', alpha=0.1, label='Simulated RH gradient' if i == 0 else None, zorder=1)
plt.xlabel('RH Gradient (%/m)')
plt.ylabel('Altitude (m)')
plt.title('Finite Difference vs Smoothed RH Gradient')
plt.legend(loc='upper left')
plt.grid(True)
#plt.xlim(-1.5, 1.5)
#plt.xlim(-2, 2)

plt.subplot(1, 3, 3)

plt.hist(simulations_pblh_rh, bins=10, alpha=0.7, color='steelblue', edgecolor='black', label='MCM PBLH counts')
plt.axvline(pblh_rh, color=map_labels_to_colors['rh'], linestyle='--', linewidth=2, label=f'PBLH (Observations): {pblh_rh:.1f} m')
plt.axvline(sm_pblh_rh, color=map_labels_to_colors['wspeed'], linestyle=':', linewidth=2, label=f'PBLH (Smoothed): {sm_pblh_rh:.1f} m')
plt.axvline(simulations_pblh_rh_avg, color='gray', linestyle='--', linewidth=2, label=f'PBLH (MCM): {simulations_pblh_rh_avg:.1f} m ± {simulations_pblh_rh_unc:.1f} m')
plt.xlabel('PBLH (m)')
plt.ylabel('Count')
plt.ylim(top=nsimulations)
plt.title('MCM PBLH Histogram')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.show()

raise

plt.figure(figsize=(10, 12))
plt.suptitle(f'RS41-GDP: {where}, {when}'#, {file_index}'
            , fontsize=20)

plt.subplot(1, 2, 1)
plt.scatter(seconds, altitude, marker='o', s=4, linestyle='-', label='Observed altitude', color=map_labels_to_colors['alt'])
plt.fill_between(
    seconds,
    altitude - altitude_unc,
    altitude + altitude_unc,
    color=map_labels_to_colors['alt_uc'],
    alpha=0.3,
    label='Altitude uncertainty',
)
plt.scatter(seconds, smoothed_altitude, marker='o', s=4, label='Smoothed altitude', color=map_labels_to_colors['temp'])
plt.fill_between(
    seconds,
    smoothed_altitude - smoother_altitude_unc,
    smoothed_altitude + smoother_altitude_unc,
    color=map_labels_to_colors['temp_uc'],
    alpha=0.3,
    label='Smoothed altitude uncertainty',
)
for i in range(nsimulations):
    plt.plot(seconds, simulations[i][0], color='gray', alpha=0.2, label='Simulated altitude' if i == 0 else None)
plt.xlabel('Seconds since start')
plt.ylabel('Altitude (m)')
plt.title('Observed vs Smoothed Altitude')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)

plt.scatter(seconds[:-1], diff_ratio_alt, marker='o', s=4, label='Finite difference altitude velocity', color=map_labels_to_colors['alt'])
plt.fill_between(
    seconds[:-1],
    diff_ratio_alt - diff_ratio_alt_unc,
    diff_ratio_alt + diff_ratio_alt_unc,
    color=map_labels_to_colors['alt_uc'],
    alpha=0.3,
    label='Finite difference altitude velocity uncertainty',
)
plt.scatter(seconds, smoothed_alt_vel, marker='o', s=4, label='Smoothed altitude velocity', color=map_labels_to_colors['temp'])
plt.fill_between(
    seconds,
    smoothed_alt_vel - smoother_alt_vel_unc,
    smoothed_alt_vel + smoother_alt_vel_unc,
    color=map_labels_to_colors['temp_uc'],
    alpha=0.3,
    label='Smoothed altitude velocity uncertainty',
)
for i in range(nsimulations):
    plt.plot(seconds, simulations[i][1], color='gray', alpha=0.2, label='Simulated altitude velocity' if i == 0 else None)
plt.xlabel('Seconds since start')
plt.ylabel('Altitude Velocity (m/s)')
plt.title('Finite Difference vs Smoothed Altitude Velocity')
plt.legend()
plt.grid(True)

plt.show()

# PLOT 2

plt.figure(figsize=(10, 12))
plt.suptitle(f'RS41-GDP: {where}, {when}'#, {file_index}'
            , fontsize=20)

plt.subplot(1, 2, 1)

plt.scatter(seconds, rh, marker='o', s=4, linestyle='-', label='Observed RH', color=map_labels_to_colors['rh'])
plt.fill_between(
    seconds,
    rh - rh_unc,
    rh + rh_unc,
    color=map_labels_to_colors['rh_uc'],
    alpha=0.3,
    label='RH uncertainty',
)
plt.scatter(seconds, smoothed_rh, marker='o', s=4, label='Smoothed RH', color=map_labels_to_colors['wspeed'])
plt.fill_between(
    seconds,
    smoothed_rh - smoother_rh_unc,
    smoothed_rh + smoother_rh_unc,
    color=map_labels_to_colors['wspeed_uc'],
    alpha=0.3,
    label='Smoothed RH uncertainty',
)
for i in range(nsimulations):
    plt.plot(seconds, simulations[i][2], color='gray', alpha=0.2, label='Simulated RH' if i == 0 else None)
plt.xlabel('Seconds since start')
plt.ylabel('RH (%)')
plt.title('Observed RH')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)


plt.scatter(seconds[:-1], diff_ratio_rh, marker='o', s=4, label='Finite difference RH velocity', color=map_labels_to_colors['rh'])
plt.fill_between(
    seconds[:-1],
    diff_ratio_rh - diff_ratio_rh_unc,
    diff_ratio_rh + diff_ratio_rh_unc,
    color=map_labels_to_colors['rh_uc'],
    alpha=0.3,
    label='Finite difference RH velocity uncertainty',
)
plt.scatter(seconds, smoothed_rh_vel, marker='o', s=4, label='Smoothed RH velocity', color=map_labels_to_colors['wspeed'])
plt.fill_between(
    seconds,
    smoothed_rh_vel - smoother_rh_vel_unc,
    smoothed_rh_vel + smoother_rh_vel_unc,
    color=map_labels_to_colors['wspeed_uc'],
    alpha=0.3,
    label='Smoothed RH velocity uncertainty',
)
for i in range(nsimulations):
    plt.plot(seconds, simulations[i][3], color='gray', alpha=0.2, label='Simulated RH velocity' if i == 0 else None)
plt.xlabel('Seconds since start')
plt.ylabel('RH Velocity (%/s)')
plt.title('Finite Difference vs Smoothed RH Velocity')
plt.legend()
plt.grid(True)


plt.show()

# PLOT 3

plt.figure(figsize=(10, 12))
plt.suptitle(f'RS41-GDP: {where}, {when}'#, {file_index}'
            , fontsize=20)

plt.subplot(1, 2, 1)

plt.plot(rh, altitude, label='Observed Vertical Profile', color=map_labels_to_colors['rh'])
plt.plot(smoothed_rh, smoothed_altitude, label='Smoothed Vertical Profile', color=map_labels_to_colors['wspeed'])
for i in range(nsimulations):
    plt.plot(simulations[i][2], simulations[i][0], color='gray', alpha=0.2, label='Simulated Vertical Profile' if i == 0 else None)
plt.fill_betweenx(
    altitude,
    rh - rh_unc,
    rh + rh_unc,
    color=map_labels_to_colors['rh_uc'],
    alpha=0.3,
    label='RH uncertainty',
)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Altitude (m)')
plt.title('Vertical Profile with PBLH Estimates')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)

plt.scatter(seconds[:-1], diff_grad_rh, marker='o', s=4, label='Finite difference RH gradient', color=map_labels_to_colors['rh'])
plt.fill_between(
    seconds[:-1],
    diff_grad_rh - diff_grad_rh_unc,
    diff_grad_rh + diff_grad_rh_unc,
    color=map_labels_to_colors['rh_uc'],
    alpha=0.3,
    label='Finite difference RH gradient uncertainty',
)
plt.scatter(seconds, smoothed_grad_rh, marker='o', s=4, label='Smoothed RH gradient', color=map_labels_to_colors['wspeed'])
plt.fill_between(
    seconds,
    smoothed_grad_rh - smoothed_grad_rh_unc,
    smoothed_grad_rh + smoothed_grad_rh_unc,
    color=map_labels_to_colors['wspeed_uc'],
    alpha=0.3,
    label='Smoothed RH gradient uncertainty',
)
for i in range(nsimulations):
    plt.plot(seconds, simulations[i][4], color='gray', alpha=0.2, label='Simulated RH gradient' if i == 0 else None)
plt.xlabel('Seconds since start')
plt.ylabel('RH Gradient (%/m)')
plt.title('Finite Difference vs Smoothed RH Gradient')
plt.legend()
plt.grid(True)

plt.show()

# PLOT 4

plt.figure(figsize=(10, 12))
plt.suptitle(f'RS41-GDP: {where}, {when}'#, {file_index}'
            , fontsize=20)

plt.subplot(1, 2, 1)

plt.plot(rh, altitude, label='Observed Vertical Profile', color=map_labels_to_colors['rh'])

plt.plot(smoothed_rh, smoothed_altitude, label='Smoothed Vertical Profile', color=map_labels_to_colors['wspeed'])
if pblh_rh is not None:
    plt.axhline(pblh_rh, color=map_labels_to_colors['rh'], linestyle='--', label=f'PBLH (RH Gradient): {pblh_rh:.1f} m')
if sm_pblh_rh is not None:
    plt.axhline(sm_pblh_rh, color=map_labels_to_colors['wspeed'], linestyle='--', label=f'PBLH (Smoothed RH Gradient): {sm_pblh_rh:.1f} m')
if simulations_pblh_rh_avg is not None:
    plt.axhline(simulations_pblh_rh_avg, color='gray', linestyle='--', label=f'PBLH (Simulated RH Gradient): {simulations_pblh_rh_avg:.1f} m ± {simulations_pblh_rh_unc:.1f} m')
if simulations_pblh_rh_unc is not None:
    plt.fill_betweenx(
        y=[simulations_pblh_rh_avg - simulations_pblh_rh_unc, simulations_pblh_rh_avg + simulations_pblh_rh_unc],
        x1=0,
        x2=100,
        color='gray',
        alpha=0.1,
        label='Simulated PBLH uncertainty',
    )
plt.fill_betweenx(
    altitude,
    rh - rh_unc,
    rh + rh_unc,
    color=map_labels_to_colors['rh_uc'],
    alpha=0.3,
    label='RH uncertainty',
)
for i in range(nsimulations):
    plt.plot(simulations[i][2], simulations[i][0], color='gray', alpha=0.2, label='Simulated Vertical Profile' if i == 0 else None)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Altitude (m)')
plt.title('Vertical Profile with PBLH Estimates')
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(1, 2, 2)

plt.hist(simulations_pblh_rh, bins=10, alpha=0.7, color='steelblue', edgecolor='black', label='Simulated PBLH')
plt.axvline(pblh_rh, color=map_labels_to_colors['rh'], linestyle='--', linewidth=2, label=f'PBLH (RH Gradient): {pblh_rh:.1f} m')
plt.axvline(sm_pblh_rh, color=map_labels_to_colors['wspeed'], linestyle='--', linewidth=2, label=f'PBLH (Smoothed RH Gradient): {sm_pblh_rh:.1f} m')
plt.axvline(simulations_pblh_rh_avg, color='gray', linestyle='-', linewidth=2, label=f'PBLH (Simulated Mean): {simulations_pblh_rh_avg:.1f} m ± {simulations_pblh_rh_unc:.1f} m')
plt.xlabel('PBLH (m)')
plt.ylabel('Count')
plt.ylim(top=nsimulations)
plt.title('PBLH Simulations Histogram')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.show()
