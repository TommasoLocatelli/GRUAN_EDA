import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
import os
from ssm.local_trend import RHLocalLinearTrend as RHL
import numpy as np
import matplotlib.pyplot as plt
from code_examples.visual_config.color_map import map_labels_to_colors
FOLDER = r'gdp\products_RS41-GDP-1_PAY_2024'
FOLDER = r'gdp\products_RS41-GDP-1_TEN_2024'

def pblh_preproc(gdp):
    upper_bound=gp._find_upper_bound(gdp.data[['alt']], upper_bound=3500, return_value=True) # find the PBLH upper bound for profile
    data = gdp.data[gdp.data['alt'] <= upper_bound]  # Limit to first 3.5 km
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0] # location
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0] # time
    when=when[0:10]+' '+when[11:19]
    hour = int(when[11:13])
    minutes = int(when[14:16])
    era5_times=[
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
    ]
    era5_time=min(era5_times, key=lambda x: abs(int(x[0:2]) - hour) + abs(int(x[3:5]) - minutes)/60)
    longitude = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Longitude']['Value'].values[0][:-3]
    latitude = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Latitude']['Value'].values[0][:-3]
    site_altitude = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Altitude']['Value'].values[0]

    start = data['time'].values[0]
    time = data['time'].values
    seconds = (time - start) / np.timedelta64(1, 's')
    seconds = seconds.astype(float)

    altitude = data['alt'].values
    altitude_unc = data['alt_gph_uc_ucor'].values
    altitude_var = (altitude_unc * 0.5)**2  # variance of altitude measurements

    rh = data['rh'].values
    rh_unc = data['rh_uc_ucor'].values
    rh_var = (rh_unc * 0.5)**2

    return where, when, latitude, longitude, site_altitude, era5_time, seconds, altitude, altitude_var, rh, rh_var

pblhs_results = []
for file in os.listdir(FOLDER):
    file_path = os.path.join(FOLDER, file)
    if os.path.isfile(file_path) and file.endswith('.nc'):
        gdp = gp.read(file_path)
        where, when, latitude, longitude, site_altitude, era5_time, seconds, altitude, altitude_var, rh, rh_var = pblh_preproc(gdp)
        mod = RHL(endog=np.column_stack([altitude, rh]), alt_measurement_var=altitude_var, rh_measurement_var=rh_var)
        res = mod.fit(method='lbfgs',
            maxiter=50,
            full_output=1,
            disp=5)
        
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

        nsimulations = 11
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
        simulations_pblh_rh_median = np.median(simulations_pblh_rh)
        simulations_pblh_rh_unc = np.std(simulations_pblh_rh)*2
    
    def era_blh(year, month, day, era5_time, latitude, longitude, site_altitude):
        import cdsapi
        import xarray as xr
        dataset = "reanalysis-era5-single-levels"
        ensemble_spread_request =  {
            "product_type": [
                "ensemble_spread"
            ],
            "variable": ["boundary_layer_height"],
            "year": [year],
            "month": [month],
            "day": [day],
            "time": [era5_time],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [float(latitude)+2, float(longitude)-2, float(latitude)-2, float(longitude)+2]
        }
        ensemble_mean_request = {
            "product_type": [
            "ensemble_mean"
            ],
            "variable": ["boundary_layer_height"],
            "year": [year],
            "month": [month],
            "day": [day],
            "time": [era5_time],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [float(latitude)+2, float(longitude)-2, float(latitude)-2, float(longitude)+2]
        }
        client = cdsapi.Client()
        from pathlib import Path
        target=f"temp\data\spread_{where}_{year}_{month}_{day}_{era5_time}.nc"
        path = Path(target)
        if path.exists():
            print("File already exists, skipping download.")
        else:
            client.retrieve(dataset, ensemble_spread_request).download(target=target)
        target=f"temp\data\mean_{where}_{year}_{month}_{day}_{era5_time}.nc"
        path = Path(target)
        if path.exists():
            print("File already exists, skipping download.")
        else:
            client.retrieve(dataset, ensemble_mean_request).download(target=target)
        spread = xr.open_dataset(f"temp\data\spread_{where}_{year}_{month}_{day}_{era5_time}.nc")
        mean = xr.open_dataset(f"temp\data\mean_{where}_{year}_{month}_{day}_{era5_time}.nc")
        valid_time = f"{year}-{month:02d}-{day:02d}T{era5_time}:00.000000000"
        era5_blh_mean = mean["blh"].sel(latitude=round(float(latitude)), longitude=round(float(longitude)), valid_time=valid_time).values
        era5_blh_mean+=float(site_altitude[:-2])
        era5_blh_spread = spread["blh"].sel(latitude=round(float(latitude)), longitude=round(float(longitude)), valid_time=valid_time).values
        
        return era5_blh_mean, era5_blh_spread
    
    era5_blh_mean, era5_blh_spread = era_blh(year=int(when[0:4]), month=int(when[5:7]), 
                                            day=int(when[8:10]), era5_time=era5_time, 
                                            latitude=latitude, longitude=longitude, 
                                            site_altitude=site_altitude)
    result = {
        'where': where,
        'when': when,
        'longitude': longitude,
        'latitude': latitude,
        'site_altitude': site_altitude,
        'pblh_rh': pblh_rh,
        'sm_pblh_rh': sm_pblh_rh,
        'simulations_pblh_rh_avg': simulations_pblh_rh_avg,
        'simulations_pblh_rh_median': simulations_pblh_rh_median,
        'simulations_pblh_rh_unc': simulations_pblh_rh_unc,
        'era5_blh_mean': era5_blh_mean,
        'era5_blh_spread': era5_blh_spread,
        'era5_time': era5_time,
        'era5_latitude': round(float(latitude)),
        'era5_longitude': round(float(longitude))
    }
    pblhs_results.append(result)
    if True:
        plt.figure()
        plt.suptitle(f'RS41-GDP: {where}, {when}', fontsize=20)
        plt.plot(rh, altitude, color=map_labels_to_colors['rh'], label='Measured RH')
        plt.plot(smoothed_rh, smoothed_altitude, color=map_labels_to_colors['wspeed'], label='Smoothed RH')
        for i in range(nsimulations):
            plt.plot(simulations[i][2], simulations[i][0], color='gray', alpha=0.3)
            plt.axhline(simulations_pblh_rh[i], color='black', linestyle='dotted', alpha=0.3)
        if pblh_rh is not None:
            plt.axhline(pblh_rh, color=map_labels_to_colors['rh'], linestyle='--', label=f'PBLH (RH Gradient): {pblh_rh:.1f} m')
        if sm_pblh_rh is not None:
            plt.axhline(sm_pblh_rh, color=map_labels_to_colors['wspeed'], linestyle='--', label=f'PBLH (Smoothed RH Gradient): {sm_pblh_rh:.1f} m')
        if simulations_pblh_rh_avg is not None:
            plt.axhline(simulations_pblh_rh_avg, color='gray', linestyle='--', label=f'PBLH (Simulated RH Gradient): {simulations_pblh_rh_avg:.1f} m ± {simulations_pblh_rh_unc:.1f} m')
        if simulations_pblh_rh_median is not None:
            plt.axhline(simulations_pblh_rh_median, color='black', linestyle=':', label=f'PBLH (Simulated RH Gradient Median): {simulations_pblh_rh_median:.1f} m')
        if simulations_pblh_rh_unc is not None:
            plt.fill_betweenx(
                y=[simulations_pblh_rh_avg - simulations_pblh_rh_unc, simulations_pblh_rh_avg + simulations_pblh_rh_unc],
                x1=0,
                x2=100,
                color='gray',
                alpha=0.1,
                label='Simulated PBLH uncertainty',
            )
        if era5_blh_mean is not None:
            plt.axhline(era5_blh_mean, color=map_labels_to_colors['temp'], linestyle='--', label=f'ERA5 BLH Mean: {era5_blh_mean:.1f} m ± {era5_blh_spread:.1f} m')
        if era5_blh_spread is not None:
            plt.fill_betweenx(
                y=[era5_blh_mean - era5_blh_spread, era5_blh_mean + era5_blh_spread],
                x1=0,
                x2=100,
                color=map_labels_to_colors['temp'],
                alpha=0.1,
                label='ERA5 BLH Spread',
            )
        plt.xlabel('RH')
        plt.ylabel('Altitude')
        plt.title('RH Vertical Profile')
        plt.legend()
        plt.show()

print(pblhs_results)