"""
Use a gaussian process to add noise to profiles and perform Monte Carlo simulation to estimate PBLH uncertainties.
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

M=10 # number of Monte Carlo samples

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
    for _ in tqdm(range(M)):
        data_noisy = gdp.data.copy(deep=True) # make a copy of the data to add noise to
        for var in ['temp']: ### Temperature is the only variable with Spatially Correlated uncertainty in RS-41 ###
            Z=gp.uncertainty_matrix(data_noisy, variable_clmn=var) # create uncertainty matrix for variable
            try:
                # Cholesky decomposition method
                noise = gp.gp_noise(data_noisy, Z, method='cholesky')  # generate correlated noise
                data_noisy[var] += noise # add noise to variable
            except Exception as e:
                print(f"Cholesky decomposition failed for variable {var} due to {e}. Falling back to SVD method.")
                # SVD method
                noise = gp.gp_noise(data_noisy, Z, method='svd')  # generate correlated noise
                data_noisy[var] += noise # add noise to variable

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
    # compute mean and stddev of PBLH estimates from Monte Carlo samples
    pblh_uncertainty = {
        'pm': (np.nanmean(pblh_samples['pm']), np.nanstd(pblh_samples['pm'], ddof=1)),
        'theta': (np.nanmean(pblh_samples['theta']), np.nanstd(pblh_samples['theta'],ddof=1)),
        'rh': (np.nanmean(pblh_samples['rh']), np.nanstd(pblh_samples['rh'], ddof=1)),
        'q': (np.nanmean(pblh_samples['q']), np.nanstd(pblh_samples['q'], ddof=1)),
        'Ri': (np.nanmean(pblh_samples['Ri']), np.nanstd(pblh_samples['Ri'], ddof=1)),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gdp.data['temp'], gdp.data['alt'], 'k-', linewidth=2, label='Original')
    for i, profile in enumerate(noisy_profiles):
        ax.plot(profile['temp'], profile['alt'], alpha=0.8, linewidth=0.5)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Temperature Profiles - {where} ({when})')
    ax.legend(['Original'] + [f'MC Sample {i+1}' for i in range(min(5, len(noisy_profiles)))])
    ax.grid(True, alpha=0.3)

    if pblh_pm is not None: # plot PBLH line
        ax.axhline(pblh_pm, color=map_labels_to_colors['pblh_pm'], linestyle=':', linewidth=2, label=f'PM PBLH: {pblh_pm:.1f} m')

    if pblh_theta is not None: # plot PBLH line
        ax.axhline(pblh_theta, color=map_labels_to_colors['pblh_theta'], linestyle=':', linewidth=2, label=f'THETA PBLH: {pblh_theta:.1f} m')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (m)')
    ax.grid()
    # Add PBLH lines and uncertainty bands
    x_bounds=[ax.get_xlim()[0],ax.get_xlim()[1]]
    for label, (mean, std) in pblh_uncertainty.items():
        temperature_methods=['pm', 'theta']
        if mean is not None and label in temperature_methods:
            ax.axhline(mean, linestyle='--', label=f'{label.upper()} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
            ax.fill_betweenx(
                y=np.array([mean - std*10, mean + std*10]),
                x1=x_bounds[0],
                x2=x_bounds[1],
                color=map_labels_to_colors['pblh_'+label],
                alpha=0.1,
                label=f'{label.upper()} MC Uncertainty: {std:.1f} m'
            )
    plt.tight_layout()
    plt.show()

    #break # Remove this break to process all files