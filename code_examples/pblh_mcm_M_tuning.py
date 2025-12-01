"""
Code used to generate poster plots for ICM16 conference 2025.
Example script to analyze and visualize the uncertainty in PBLH estimation using a naive Monte Carlo approach.
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
from visual_config.color_map import map_labels_to_colors
import seaborn as sns
import pandas as pd
from tqdm import tqdm

VERTICAL_PROFILE_PLOT = True # The main plot
ADD_VIRTUAL_TEMPERATURE =  True # Add virtual temp in temp profile
ADD_SPECIFIC_HUMIDITY = True # Add specific humidity profile
ADD_PM = True # add Parcel Method (new with respect to the poster)
ADD_Q = True # add specific humidity method
INCLUDE_RI = False # Include bulk richardon number profile
M = 250 # number of Monte Carlo samples

uncertainty='_uc' # '_uc_ucor' or '_uc' choose uncertainty type

noise_function = gp.gaussian_noise # no autocorrelation or crosscorrelation

folder = r'gdp\icm16' # open folder with chosen GDP files
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:]:
    file_index = file_paths.index(file_path)
    gdp = gp.read(file_path) # read GDP file
    upper_bound=gp._find_upper_bound(gdp.data[['alt']], upper_bound=3500, return_value=True) # find the PBLG upper bound for profile
    gdp.data = gdp.data[gdp.data['alt'] <= upper_bound]  # Limit to first 3.5 km
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0] # location
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0] # time

    # calculate PBLH using different methods to original data
    # all methods add columns to gdp.data with results and intermediate variables
    data = gp.parcel_method(gdp.data) # calculate PBLH using parcel method
    data = gp.potential_temperature_gradient(gdp.data, virtual=True) # calculate potential temperature gradient
    data = gp.RH_gradient(gdp.data) # calculate RH gradient
    data = gp.specific_humidity_gradient(gdp.data) # calculate specific humidity gradient
    data = gp.bulk_richardson_number_method(gdp.data) # calculate PBLH using bulk Richardson number method
    # extract PBLH estimates
    pblh_pm = gdp.data['alt'][data['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data and any(data['pblh_pm'] == 1) else None
    pblh_theta = gdp.data['alt'][data['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data and any(data['pblh_theta'] == 1) else None 
    pblh_rh = gdp.data['alt'][data['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data and any(data['pblh_rh'] == 1) else None
    pblh_q = gdp.data['alt'][data['pblh_q'] == 1].iloc[0] if 'pblh_q' in data and any(data['pblh_q'] == 1) else None
    pblh_Ri = gdp.data['alt'][data['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data and any(data['pblh_Ri'] == 1) else None
    
    # Monte Carlo simulation
    noisy_profiles = []
    pblh_samples = {'pm': [], 'theta': [], 'rh': [], 'q': [], 'Ri': []}
    noise_coeff = 0.5 # divide by k=2 to regain the standard combined uncertainty
    for _ in tqdm(range(M)):
        data_noisy = gdp.data.copy(deep=True) # make a copy of the data to add noise to
        data_noisy['alt'] = noise_function(data_noisy['alt'], data_noisy['alt_uc']*noise_coeff) # add noise to altitude
        data_noisy=data_noisy.sort_values('alt').reset_index(drop=True) # sort by altitude after noise addition
        data_noisy['temp'] = noise_function(data_noisy['temp'], data_noisy['temp'+uncertainty]*noise_coeff) # add noise to temperature
        data_noisy['rh'] = noise_function(data_noisy['rh'], data_noisy['rh'+uncertainty]*noise_coeff) # add noise to RH
        data_noisy['press'] = noise_function(data_noisy['press'], data_noisy['press_uc']*noise_coeff) # add noise to pressure
        data_noisy['wspeed'] = noise_function(data_noisy['wspeed'], data_noisy['wspeed'+uncertainty]*noise_coeff) # add noise to wind speed
        data_noisy['wdir'] = noise_function(data_noisy['wdir'], data_noisy['wdir'+uncertainty]*noise_coeff) # add noise to wind direction
        data_noisy = gp.parcel_method(data_noisy) # calculate PBLH using parcel method
        data_noisy = gp.potential_temperature_gradient(data_noisy, virtual=True) # calculate potential temperature gradient
        data_noisy = gp.RH_gradient(data_noisy) # calculate RH gradient
        data_noisy = gp.bulk_richardson_number_method(data_noisy) # calculate PBLH using bulk Richardson number method
        noisy_profiles.append(data_noisy) # store noisy profile
        # extract PBLH estimates from noisy profile
        pblh_samples['pm'].append(data_noisy['alt'][data_noisy['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data_noisy and any(data_noisy['pblh_pm'] == 1) else None)
        pblh_samples['theta'].append(data_noisy['alt'][data_noisy['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data_noisy and any(data_noisy['pblh_theta'] == 1) else None)
        pblh_samples['rh'].append(data_noisy['alt'][data_noisy['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data_noisy and any(data_noisy['pblh_rh'] == 1) else None)
        pblh_samples['q'].append(data_noisy['alt'][data_noisy['pblh_q'] == 1].iloc[0] if 'pblh_q' in data_noisy and any(data_noisy['pblh_q'] == 1) else 0)
        pblh_samples['Ri'].append(data_noisy['alt'][data_noisy['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data_noisy and any(data_noisy['pblh_Ri'] == 1) else 0)

    # compute mean and stddev of PBLH estimates with increasing M from Monte Carlo samples
    increasing_M_results = []
    for M in range(1, M+1):
        pblh_unc = {
            'pm': (np.nanmean(pblh_samples['pm'][:M]), np.nanstd(pblh_samples['pm'][:M], ddof=1)),
            'theta': (np.nanmean(pblh_samples['theta'][:M]), np.nanstd(pblh_samples['theta'][:M],ddof=1)),
            'rh': (np.nanmean(pblh_samples['rh'][:M]), np.nanstd(pblh_samples['rh'][:M], ddof=1)),
            'q': (np.nanmean(pblh_samples['q'][:M]), np.nanstd(pblh_samples['q'][:M], ddof=1)),
            'Ri': (np.nanmean(pblh_samples['Ri'][:M]), np.nanstd(pblh_samples['Ri'][:M], ddof=1)),
        }
        increasing_M_results.append(pblh_unc)
    methods = ['pm', 'theta', 'rh', 'q', 'Ri']
    plt.figure(figsize=(10, 12))

    for method in methods:
        plbhs=[result[method][0] for result in increasing_M_results]
        uncs=[result[method][1] for result in increasing_M_results]
        plt.plot(range(1, M+1), plbhs, label=f'{method} method mean PBLH', color=map_labels_to_colors['pblh_'+method])
        plt.fill_between(range(1, M+1), np.array(plbhs) - np.array(uncs), np.array(plbhs) + np.array(uncs), alpha=0.3, 
                        label=f'{method} method uncertainty', color=map_labels_to_colors['pblh_'+method])

    plt.xlabel('Number of Monte Carlo Samples (M)')
    plt.ylabel('PBLH Estimate and Uncertainty (m)')
    plt.title(f'Convergence of PBLH Estimates and Uncertainties with Increasing M\nLocation: {where}, Time: {when}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    break # Remove this break to process all files