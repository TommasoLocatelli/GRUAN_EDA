import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
from visual_config.color_map import map_labels_to_colors

noise_functions = [gp.gaussian_noise, gp.smooth_noise]
noise_function = noise_functions[1]  # no autocorrelation or crosscorrelation

folder = r'gdp\icm16'
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:5]:
    gdp = gp.read(file_path)
    upper_bound=gp._find_upper_bound(gdp.data[['alt']], upper_bound=5000, retnrn_value=True)
    gdp.data = gdp.data[gdp.data['alt'] <= upper_bound]  # Limit to first 5 km
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
    
    # Monte Carlo uncertainty estimation
    n_samples = 100
    noisy_profiles = []
    pblh_samples = {'pm': [], 'theta': [], 'rh': [], 'Ri': []}
    noise_coeff = 0.5 # divide by k=2 to regain the standard combined uncertainty
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
    plt.figure(figsize=(10, 12))
    plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)

    # Temperature plot
    ax1 = plt.subplot(1, 3, 1)
    for sample in noisy_profiles:
        ax1.plot(sample['temp'], sample['alt'], color='gray', alpha=0.3)
        ax1.plot(sample['virtual_theta'], sample['alt'], color='gray', alpha=0.3)
    ax1.scatter(gdp.data['temp'], gdp.data['alt'], label='True Temperature', color=map_labels_to_colors['temp'], linewidth=2, zorder=5)
    ax1.scatter(gdp.data['virtual_theta'], gdp.data['alt'], label='True Virtual Potential Temperature', color=map_labels_to_colors['virtual_theta'], linewidth=2, zorder=5)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Altitude (m)')
    ax1.grid()

    # Add PBLH lines and uncertainty bands
    for label, (mean, std) in pblh_uncertainty.items():
        if mean is not None and label in ['pm', 'theta']:
            ax1.axhline(mean, linestyle='--', label=f'{label} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
            ax1.fill_betweenx(
                y=np.array([mean - std, mean + std]),
                x1=min(gdp.data['temp'].min()-1, gdp.data['virtual_theta'].min()-1),
                x2=max(gdp.data['temp'].max()+1, gdp.data['virtual_theta'].max()+1),
                color=map_labels_to_colors['pblh_'+label],
                alpha=0.1,
                label=f'{label} MC Uncertainty: {std:.1f} m'
            )
    plt.legend(loc='best')

    # RH plot
    ax2 = plt.subplot(1, 3, 2)
    for sample in noisy_profiles:
        ax2.plot(sample['rh'], sample['alt'], color='gray', alpha=0.3)
    ax2.scatter(gdp.data['rh'], gdp.data['alt'], label='True RH', color=map_labels_to_colors['rh'], linewidth=2, zorder=5)
    ax2.set_xlabel('Relative Humidity (%)')
    ax2.set_ylabel('Altitude (m)')
    ax2.grid()

    # Add PBLH lines and uncertainty bands
    for label, (mean, std) in pblh_uncertainty.items():
        if mean is not None and label in ['rh']:
            ax2.axhline(mean, linestyle='--', label=f'{label} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
            ax2.fill_betweenx(
                y=np.array([mean - std, mean + std]),
                x1=gdp.data['rh'].min()-1,
                x2=gdp.data['rh'].max()+1,
                color=map_labels_to_colors['pblh_'+label],
                alpha=0.1,
                label=f'{label} MC Uncertainty: {std:.1f} m'
            )
    plt.legend(loc='best')

    # wind speed plot
    ax3 = plt.subplot(1, 3, 3)
    for sample in noisy_profiles:
        ax3.plot(sample['wspeed'], sample['alt'], color='gray', alpha=0.3)
    ax3.scatter(gdp.data['wspeed'], gdp.data['alt'], label='True Wind Speed', color=map_labels_to_colors['wspeed'], linewidth=2, zorder=5)
    ax3.set_xlabel('Wind Speed (m/s)')
    ax3.set_ylabel('Altitude (m)')
    ax3.grid()

    # Add PBLH lines and uncertainty bands
    for label, (mean, std) in pblh_uncertainty.items():
        if mean is not None and label in ['Ri']:
            ax3.axhline(mean, linestyle='--', label=f'{label} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
            ax3.fill_betweenx(
                y=np.array([mean - std, mean + std]),
                x1=gdp.data['wspeed'].min()-1,
                x2=gdp.data['wspeed'].max()+1,
                color=map_labels_to_colors['pblh_'+label],
                alpha=0.1,
                label=f'{label} MC Uncertainty: {std:.1f} m'
            )

    plt.legend(loc='best')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit suptitle
    plt.show()
    break  # Remove this break to process all files