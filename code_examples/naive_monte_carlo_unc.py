import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp

noise_functions = [gp.gaussian_noise, gp.smooth_noise] # how to conserve dispersion with correlated noise?
noise_function = noise_functions[0]  # Choose the noise function to use

folder = r'gdp\products_RS41-GDP-1_POT_2025'
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:5]:
    gdp = gp.read(file_path)
    gdp.data = gdp.data[gdp.data['alt'] <= 2000]  # Limit to first 10 km for speed
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]

    data = gp.parcel_method(gdp.data)
    data = gp.potential_temperature_gradient(gdp.data, virtual=True)
    data = gp.RH_gradient(gdp.data)
    data = gp.bulk_richardson_number_method(gdp.data)
    pblh_pm = gdp.data['alt'][data['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data and any(data['pblh_pm'] == 1) else None
    pblh_theta = gdp.data['alt'][data['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data and any(data['pblh_theta'] == 1) else None
    pblh_rh = gdp.data['alt'][data['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data and any(data['pblh_rh'] == 1) else None
    pblh_Ri = gdp.data['alt'][data['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data and any(data['pblh_Ri'] == 1) else None
    
    # Naive Monte Carlo uncertainty estimation
    n_samples = 100
    noisy_profiles = []
    pblh_samples = {'pm': [], 'theta': [], 'rh': [], 'Ri': []}
    noise_coeff = 0.5# 0.005  # fraction of the uncertainty to use as stddev for noise
    for _ in range(n_samples):
        data_noisy = gdp.data.copy()
        data_noisy['alt'] = noise_function(data_noisy['alt'], data_noisy['alt_uc']*noise_coeff)
        data_noisy=data_noisy.sort_values('alt').reset_index(drop=True)
        data_noisy['temp'] = noise_function(data_noisy['temp'], data_noisy['temp_uc']*noise_coeff)
        data_noisy['rh'] = noise_function(data_noisy['rh'], data_noisy['rh_uc']*noise_coeff)
        data_noisy['press'] = noise_function(data_noisy['press'], data_noisy['press_uc']*noise_coeff)
        data_noisy = gp.parcel_method(data_noisy)
        data_noisy = gp.potential_temperature_gradient(data_noisy, virtual=True)
        data_noisy = gp.RH_gradient(data_noisy)
        data_noisy = gp.bulk_richardson_number_method(data_noisy)
        noisy_profiles.append(data_noisy)
        pblh_samples['pm'].append(data_noisy['alt'][data_noisy['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data_noisy and any(data_noisy['pblh_pm'] == 1) else None)
        pblh_samples['theta'].append(data_noisy['alt'][data_noisy['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data_noisy and any(data_noisy['pblh_theta'] == 1) else None)
        pblh_samples['rh'].append(data_noisy['alt'][data_noisy['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data_noisy and any(data_noisy['pblh_rh'] == 1) else None)
        pblh_samples['Ri'].append(data_noisy['alt'][data_noisy['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data_noisy and any(data_noisy['pblh_Ri'] == 1) else None)
    pblh_uncertainty = {
        'pm': (np.nanmean(pblh_samples['pm']), np.nanstd(pblh_samples['pm'])),
        'theta': (np.nanmean(pblh_samples['theta']), np.nanstd(pblh_samples['theta'])),
        'rh': (np.nanmean(pblh_samples['rh']), np.nanstd(pblh_samples['rh'])),
        'Ri': (np.nanmean(pblh_samples['Ri']), np.nanstd(pblh_samples['Ri'])),
    }
    
    # plot noisy profiles with pblh estimates and uncertainties
    fig, ax = plt.subplots(figsize=(6, 8))
    for sample in noisy_profiles:
        ax.plot(sample['temp'], sample['alt'], color='gray', alpha=0.3)
    ax.scatter(gdp.data['temp'], gdp.data['alt'], label='True Temperature', color="#781E1A", linewidth=2 , zorder=5)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Altitude (m)')
    ax.axhline(pblh_pm, color='r', linestyle=':', label=f'Parcel Method PBLH: {pblh_pm:.1f} m')
    ax.axhline(pblh_theta, color='g', linestyle=':', label=f'Theta Gradient PBLH: {pblh_theta:.1f} m')
    ax.axhline(pblh_rh, color='b', linestyle=':', label=f'RH Gradient PBLH: {pblh_rh:.1f} m')
    ax.axhline(pblh_Ri, color='m', linestyle=':', label=f'Bulk Ri PBLH: {pblh_Ri:.1f} m')
    ax.set_title(f'Temperature Profiles with Noise\nSite: {where}, Time: {when}')
    ax.grid()

    # Plot mean PBLH estimates
    colors={'pm': 'r', 'theta': 'g', 'rh': 'b', 'Ri': 'm'}
    for label, (mean, std) in pblh_uncertainty.items():
        if mean is not None:
            ax.axhline(mean, linestyle='--', label=f'{label} MC PBLH: {mean:.1f} m', color=colors[label])
            ax.fill_betweenx(
            y=np.array([mean - std, mean + std]),
            x1=gdp.data['temp'].min()-1,
            x2=gdp.data['temp'].max()+1,
            color=colors[label],
            alpha=0.1
        )

    plt.legend()
    plt.show()
    