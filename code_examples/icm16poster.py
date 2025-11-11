"""
Code used to generate poster for ICM16 conference 2024.
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

ADD_VIRTUAL_TEMPERATURE = False
VIOLINS_PLOT = True
VERTICAL_PROFILE_PLOT = True
INCLUDE_RI = False

noise_functions = [gp.gaussian_noise, # no autocorrelation or crosscorrelation
                   gp.smooth_noise] # some autocorrelation 
noise_function = noise_functions[1] 

folder = r'gdp\icm16' # open folder with chosen GDP files
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:5]: 
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
    data = gp.bulk_richardson_number_method(gdp.data) # calculate PBLH using bulk Richardson number method
    # extract PBLH estimates
    pblh_pm = gdp.data['alt'][data['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data and any(data['pblh_pm'] == 1) else None
    pblh_theta = gdp.data['alt'][data['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data and any(data['pblh_theta'] == 1) else None 
    pblh_rh = gdp.data['alt'][data['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data and any(data['pblh_rh'] == 1) else None
    pblh_Ri = gdp.data['alt'][data['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data and any(data['pblh_Ri'] == 1) else None
    
    # Monte Carlo simulation
    n_samples = 100
    noisy_profiles = []
    pblh_samples = {'pm': [], 'theta': [], 'rh': [], 'Ri': []}
    noise_coeff = 0.5 # divide by k=2 to regain the standard combined uncertainty
    for _ in range(n_samples):
        data_noisy = gdp.data.copy(deep=True) # make a copy of the data to add noise to
        data_noisy['alt'] = noise_function(data_noisy['alt'], data_noisy['alt_uc']*noise_coeff) # add noise to altitude
        data_noisy=data_noisy.sort_values('alt').reset_index(drop=True) # sort by altitude after noise addition
        data_noisy['temp'] = noise_function(data_noisy['temp'], data_noisy['temp_uc']*noise_coeff) # add noise to temperature
        data_noisy['rh'] = noise_function(data_noisy['rh'], data_noisy['rh_uc']*noise_coeff) # add noise to RH
        data_noisy['press'] = noise_function(data_noisy['press'], data_noisy['press_uc']*noise_coeff) # add noise to pressure
        data_noisy['wspeed'] = noise_function(data_noisy['wspeed'], data_noisy['wspeed_uc']*noise_coeff) # add noise to wind speed
        data_noisy['wdir'] = noise_function(data_noisy['wdir'], data_noisy['wdir_uc']*noise_coeff) # add noise to wind direction
        data_noisy = gp.parcel_method(data_noisy) # calculate PBLH using parcel method
        data_noisy = gp.potential_temperature_gradient(data_noisy, virtual=True) # calculate potential temperature gradient
        data_noisy = gp.RH_gradient(data_noisy) # calculate RH gradient
        data_noisy = gp.bulk_richardson_number_method(data_noisy) # calculate PBLH using bulk Richardson number method
        noisy_profiles.append(data_noisy) # store noisy profile
        # extract PBLH estimates from noisy profile
        pblh_samples['pm'].append(data_noisy['alt'][data_noisy['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data_noisy and any(data_noisy['pblh_pm'] == 1) else None)
        pblh_samples['theta'].append(data_noisy['alt'][data_noisy['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data_noisy and any(data_noisy['pblh_theta'] == 1) else None)
        pblh_samples['rh'].append(data_noisy['alt'][data_noisy['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data_noisy and any(data_noisy['pblh_rh'] == 1) else None)
        pblh_samples['Ri'].append(data_noisy['alt'][data_noisy['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data_noisy and any(data_noisy['pblh_Ri'] == 1) else 0)
    # compute mean and stddev of PBLH estimates from Monte Carlo samples
    pblh_uncertainty = {
        'pm': (np.nanmean(pblh_samples['pm']), np.nanstd(pblh_samples['pm'])),
        'theta': (np.nanmean(pblh_samples['theta']), np.nanstd(pblh_samples['theta'])),
        'rh': (np.nanmean(pblh_samples['rh']), np.nanstd(pblh_samples['rh'])),
        'Ri': (np.nanmean(pblh_samples['Ri']), np.nanstd(pblh_samples['Ri'])),
    }

    if VERTICAL_PROFILE_PLOT:
        # plot noisy profiles with pblh estimates and uncertainties
        plt.figure(figsize=(10, 12))
        plt.suptitle(f'RS41-GDP: {where}, {when}'#, {file_index}'
                    , fontsize=16)
        # Temperature plot
        ax1 = plt.subplot(1, 3+INCLUDE_RI, 1)
        for sample in noisy_profiles: # plot all noisy profiles
            ax1.plot(sample['temp'], sample['alt'], color='gray', alpha=0.2)
            if ADD_VIRTUAL_TEMPERATURE:
                ax1.plot(sample['virtual_theta'], sample['alt'], color='gray', alpha=0.2)
        ax1.plot(gdp.data['temp'], gdp.data['alt'], #label='True Temperature',
                    color=map_labels_to_colors['temp'], linewidth=2, zorder=5) # plot true temperature
        if ADD_VIRTUAL_TEMPERATURE:
            ax1.scatter(gdp.data['virtual_theta'], gdp.data['alt'], #label='True Virtual Potential Temperature', 
                    color=map_labels_to_colors['virtual_theta'], linewidth=2, zorder=5) # plot true virtual potential temperature
        if pblh_pm is not None: # plot PBLH line
            ax1.axhline(pblh_pm, color=map_labels_to_colors['pblh_pm'], linestyle=':', linewidth=2, label=f'PM PBLH: {pblh_pm:.1f} m')
        if pblh_theta is not None: # plot PBLH line
            ax1.axhline(pblh_theta, color=map_labels_to_colors['pblh_theta'], linestyle=':', linewidth=2, label=f'THETA PBLH: {pblh_theta:.1f} m')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Altitude (m)')
        ax1.grid()
        # Add PBLH lines and uncertainty bands
        for label, (mean, std) in pblh_uncertainty.items():
            if mean is not None and label in ['pm', 'theta']:
                ax1.axhline(mean, linestyle='--', label=f'{label.upper()} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
                ax1.fill_betweenx(
                    y=np.array([mean - std, mean + std]),
                    x1=ax1.get_xlim()[0],
                    x2=ax1.get_xlim()[1],
                    color=map_labels_to_colors['pblh_'+label],
                    alpha=0.1,
                    label=f'{label.upper()} MC Uncertainty: {std:.1f} m'
                )
        plt.legend(loc='upper right')

        # RH plot
        ax2 = plt.subplot(1, 3+INCLUDE_RI, 2)
        for sample in noisy_profiles: # plot all noisy profiles
            ax2.plot(sample['rh'], sample['alt'], color='gray', alpha=0.2)
        ax2.scatter(gdp.data['rh'], gdp.data['alt'], #label='True RH', 
                    color=map_labels_to_colors['rh'], linewidth=2, zorder=5) # plot true RH
        if pblh_rh is not None: # plot PBLH line
            ax2.axhline(pblh_rh, color=map_labels_to_colors['pblh_rh'], linestyle=':', linewidth=2, label=f'RH PBLH: {pblh_rh:.1f} m')
        ax2.set_xlabel('Relative Humidity (%)')
        #ax2.set_ylabel('Altitude (m)')
        ax2.grid()
        # Add PBLH lines and uncertainty bands
        for label, (mean, std) in pblh_uncertainty.items():
            if mean is not None and label in ['rh']:
                ax2.axhline(mean, linestyle='--', label=f'{label.upper()} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
                ax2.fill_betweenx(
                    y=np.array([mean - std, mean + std]),
                    x1=ax2.get_xlim()[0],
                    x2=ax2.get_xlim()[1],
                    color=map_labels_to_colors['pblh_'+label],
                    alpha=0.1,
                    label=f'{label.upper()} MC Uncertainty: {std:.1f} m'
                )
        plt.legend(loc='upper right')

        # wind speed plot
        ax3 = plt.subplot(1, 3+INCLUDE_RI, 3)
        for sample in noisy_profiles: # plot all noisy profiles
            ax3.plot(sample['wspeed'], sample['alt'], color='gray', alpha=0.2)
        ax3.scatter(gdp.data['wspeed'], gdp.data['alt'], #label='True Wind Speed', 
                    color=map_labels_to_colors['wspeed'], linewidth=2, zorder=5) # plot true wind speed
        if pblh_Ri is not None: # plot PBLH line
            ax3.axhline(pblh_Ri, color=map_labels_to_colors['pblh_Ri'], linestyle=':', linewidth=2, label=f'Ri PBLH: {pblh_Ri:.1f} m')
        ax3.set_xlabel('Wind Speed (m/s)')
        #ax3.set_ylabel('Altitude (m)')
        ax3.grid()
        # Add PBLH lines and uncertainty bands
        for label, (mean, std) in pblh_uncertainty.items():
            if mean is not None and label in ['Ri']:
                ax3.axhline(mean, linestyle='--', label=f'{label} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
                ax3.fill_betweenx(
                    y=np.array([mean - std, mean + std]),
                    x1=ax3.get_xlim()[0],
                    x2=ax3.get_xlim()[1],
                    color=map_labels_to_colors['pblh_'+label],
                    alpha=0.1,
                    label=f'{label.upper()} MC Uncertainty: {std:.1f} m'
                )
        plt.legend(loc='upper right')

        if False:
            # plot temp gradients
            ax4 = plt.subplot(2, 3, 4)
            for sample in noisy_profiles: # plot all noisy profiles
                ax4.plot(sample['theta_gradient'], sample['alt'], color='gray', alpha=0.2)
            ax4.scatter(data['theta_gradient'], data['alt'], #label='Theta Gradient', 
                        color=map_labels_to_colors['theta'], linewidth=2, zorder=5) # plot theta gradient
            if pblh_theta is not None: # plot PBLH line
                ax4.axhline(pblh_theta, color=map_labels_to_colors['pblh_theta'], linestyle=':', linewidth=2, label=f'Theta PBLH: {pblh_theta:.1f} m')
            ax4.set_xlabel('Theta Gradient (K/m)')
            ax4.set_ylabel('Altitude (m)')
            ax4.grid()
            # Add PBLH lines and uncertainty bands
            for label, (mean, std) in pblh_uncertainty.items():
                if mean is not None and label in ['theta']:
                    ax4.axhline(mean, linestyle='--', label=f'{label} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
                    ax4.fill_betweenx(
                        y=np.array([mean - std, mean + std]),
                        x1=ax4.get_xlim()[0],
                        x2=ax4.get_xlim()[1],
                        color=map_labels_to_colors['pblh_'+label],
                        alpha=0.1,
                        label=f'{label} MC Uncertainty: {std:.1f} m'
                    )
            plt.legend(loc='upper right')

            #plot rh gradients
            ax5 = plt.subplot(2, 3, 5)
            for sample in noisy_profiles: # plot all noisy profiles
                ax5.plot(sample['rh_gradient'], sample['alt'], color='gray', alpha=0.2)
            ax5.scatter(data['rh_gradient'], data['alt'], #label='RH Gradient', 
                        color=map_labels_to_colors['rh'], linewidth=2, zorder=5) # plot rh gradient
            if pblh_rh is not None: # plot PBLH line
                ax5.axhline(pblh_rh, color=map_labels_to_colors['pblh_rh'], linestyle=':', linewidth=2, label=f'RH PBLH: {pblh_rh:.1f} m')
            ax5.set_xlabel('RH Gradient (%/m)')
            ax5.set_ylabel('Altitude (m)')
            ax5.grid()
            # Add PBLH lines and uncertainty bands
            for label, (mean, std) in pblh_uncertainty.items():
                if mean is not None and label in ['rh']:
                    ax5.axhline(mean, linestyle='--', label=f'{label} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
                    ax5.fill_betweenx(
                        y=np.array([mean - std, mean + std]),
                        x1=ax5.get_xlim()[0],
                        x2=ax5.get_xlim()[1],
                        color=map_labels_to_colors['pblh_'+label],
                        alpha=0.1,
                        label=f'{label} MC Uncertainty: {std:.1f} m'
                    )
            plt.legend(loc='upper right')

        if INCLUDE_RI:
            #plot Ri profile
            ax6 = plt.subplot(1, 4, 4)
            for sample in noisy_profiles: # plot all noisy profiles
                ax6.plot(sample['Ri_b'], sample['alt'], color='gray', alpha=0.2)
            ax6.scatter(data['Ri_b'], data['alt'], #label='Bulk Richardson Number', 
                        color=map_labels_to_colors['Ri_b'], linewidth=2, zorder=5) # plot Ri profile
            if pblh_Ri is not None: # plot PBLH line
                ax6.axhline(pblh_Ri, color=map_labels_to_colors['pblh_Ri'], linestyle=':', linewidth=2, label=f'Ri PBLH: {pblh_Ri:.1f} m')
            ax6.set_xlabel('Bulk Richardson Number')
            #ax6.set_ylabel('Altitude (m)')
            ax6.grid()
            # Add PBLH lines and uncertainty bands
            for label, (mean, std) in pblh_uncertainty.items():
                if mean is not None and label in ['Ri']:
                    ax6.axhline(mean, linestyle='--', label=f'{label} MC PBLH: {mean:.1f} m', color=map_labels_to_colors['pblh_'+label])
                    ax6.fill_betweenx(
                        y=np.array([mean - std, mean + std]),
                        x1=ax6.get_xlim()[0],
                        x2=ax6.get_xlim()[1],
                        color=map_labels_to_colors['pblh_'+label],
                        alpha=0.1,
                        label=f'{label.upper()} MC Uncertainty: {std:.1f} m'
                    )
            plt.legend(loc='upper right')

        plt.show()

    if VIOLINS_PLOT:
        # enhanced violin plot of PBLH estimates from Monte Carlo samples
        methods = list(pblh_samples.keys())
        rows = []
        colors = []
        for m in methods:
            vals = np.array([v for v in pblh_samples[m] if v is not None and not np.isnan(v)])
            if vals.size > 0:
                for v in vals:
                    rows.append((m.upper(), float(v)))
                colors.append(map_labels_to_colors.get('pblh_'+m, 'k'))

        if not rows:
            print("No valid PBLH samples to plot.")
        else:
            # try seaborn for nicer violins, fallback to matplotlib
            try:
                df = pd.DataFrame(rows, columns=['method', 'pblh'])
                order = sorted(df['method'].unique(), key=lambda x: methods.index(x.lower()))
                palette = {m.upper(): map_labels_to_colors.get('pblh_'+m, 'k') for m in methods if m.upper() in order}
                plt.figure(figsize=(8, 4))
                ax = sns.violinplot(x='method', y='pblh', data=df, order=order, palette=palette,
                                    inner=None, cut=0, scale='width')
                # overlay quartiles and median
                sns.boxplot(x='method', y='pblh', data=df, order=order, showcaps=True,
                            boxprops={'facecolor':'none', "zorder":10}, showfliers=False,
                            whiskerprops={'linewidth':1}, width=0.15)
                # overlay jittered points
                sns.stripplot(x='method', y='pblh', data=df, order=order, color='k',
                            size=3, jitter=0.15, alpha=0.6)
                ax.set_ylabel('PBLH (m)')
                ax.set_title('PBLH distribution across methods in Monte Carlo samples')
                ax.grid(axis='y', linestyle='--', alpha=0.4)
                plt.tight_layout()
                plt.show()
            except Exception:
                # matplotlib fallback
                grouped = {}
                labels = []
                data_by_method = []
                for m in methods:
                    vals = np.array([v for v in pblh_samples[m] if v is not None and not np.isnan(v)])
                    if vals.size > 0:
                        labels.append(m.upper())
                        data_by_method.append(vals)
                        grouped[m.upper()] = vals
                plt.figure(figsize=(8, 4))
                ax = plt.gca()
                parts = ax.violinplot(data_by_method, positions=np.arange(1, len(data_by_method)+1),
                                    showmeans=False, showmedians=False, showextrema=False)
                # color violins
                for pc, c in zip(parts['bodies'], colors):
                    pc.set_facecolor(c)
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.45)
                # overlay medians and IQR
                for i, vals in enumerate(data_by_method, start=1):
                    q1, m, q3 = np.percentile(vals, [25, 50, 75])
                    ax.plot([i-0.15, i+0.15], [m, m], color='white', linewidth=2)
                    ax.vlines(i, q1, q3, color='k', linewidth=2)
                    # jittered points
                    x = np.random.normal(i, 0.06, size=vals.size)
                    ax.scatter(x, vals, color=colors[i-1], edgecolor='k', linewidth=0.2, s=20, alpha=0.6)
                ax.set_xticks(np.arange(1, len(labels)+1))
                ax.set_xticklabels(labels)
                ax.set_ylabel('PBLH (m)')
                ax.set_title('PBLH distribution across methods in Monte Carlo samples')
                ax.grid(axis='y', linestyle='--', alpha=0.4)
                plt.tight_layout()
                plt.show()

        #break # Remove this break to process all files