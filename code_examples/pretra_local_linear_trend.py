import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
from code_examples.visual_config.color_map import map_labels_to_colors
from ssm.statsmodels.pretrasformed_local_trend import PreTransformedLocalLinearTrend

example_paths = [
    r'gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc',
    r'gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc',
    r'gdp\products_RS41-GDP-1_PAY_2024\PAY-RS-01_2_RS41-GDP_001_20240109T120000_1-002-001.nc',
    r'gdp\products_RS41-GDP-1_TEN_2024\TEN-RS-01_2_RS41-GDP_001_20240103T110000_1-000-001.nc'
]

folder = r'gdp\products_RS41-GDP-1_POT-RS-01_2024'
paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.nc') and file[34:36] in ['11', '12']]
gdp=gp.read(example_paths[2])

upper_bound=gp._find_upper_bound(gdp.data[['alt']], upper_bound=3000, return_value=True) # find the PBLH upper bound for profile
data = gdp.data[gdp.data['alt'] <= upper_bound]  # Limit to first 3.5 km
where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0] # location
when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0] # time
when=when[0:10]+' '+when[11:19]

start = data['time'].values[0]
time = data['time'].values
seconds = (time - start) / np.timedelta64(1, 's')
seconds = seconds.astype(float)

# -----------------------------
# ALTITUDE
# -----------------------------
z      = data['alt'].values
z_unc  = data['alt_gph_uc'].values
z_var  = (z_unc * 0.5)**2


# -----------------------------
# TEMPERATURE
# -----------------------------
T      = data['temp'].values
T_unc  = data['temp_uc'].values
T_var  = (T_unc * 0.5)**2


# -----------------------------
# PRESSURE
# -----------------------------
p      = data['press'].values
p_unc  = data['press_uc'].values
p_var  = (p_unc * 0.5)**2


# -----------------------------
# RELATIVE HUMIDITY
# -----------------------------
RH     = data['rh'].values
RH_unc = data['rh_uc'].values
RH_var = (RH_unc * 0.5)**2


# -----------------------------
# WATER-VAPOR MASS MIXING RATIO (ppm → kg/kg)
# -----------------------------
r_ppm     = data['wvmr_mass'].values
r_ppm_unc = data['wvmr_mass_uc'].values

# convert ppm → kg/kg
r         = r_ppm * 1e-6
r_unc     = r_ppm_unc * 1e-6
r_var     = (r_unc * 0.5)**2


# -----------------------------
# ZONAL WIND
# -----------------------------
u      = data['wzon'].values
u_unc  = data['wzon_uc'].values
u_var  = (u_unc * 0.5)**2


# -----------------------------
# MERIDIONAL WIND
# -----------------------------
v      = data['wmeri'].values
v_unc  = data['wmeri_uc'].values
v_var  = (v_unc * 0.5)**2


# -----------------------------
# PRE-TRASFORMATION OF VIRTUAL POTENTIAL TEMPERATURE
# -----------------------------

Thv     = gp.virtual_potential_temperature(T, p, r)
Thv_unc  = gp.virtual_potential_temperature_uncertainty(T, p, r, T_unc, p_unc, r_unc,)
Thv_var = (Thv_unc * 0.5)**2


endog=np.column_stack([z, Thv, RH, u, v])
measurement_var = np.column_stack([z_var, Thv_var, RH_var, u_var, v_var]).T
# shape: (5, T)

# Setup the model
mod = PreTransformedLocalLinearTrend(endog=endog, measurement_var=measurement_var)

# Fit it using MLE with a fixed sequence of measurement variances
res = mod.fit(method='lbfgs',
            maxiter=50,
            full_output=1,
            disp=5)
print(res.summary())

print(endog.shape)          # should be (T, 5)
print(measurement_var.shape)  # should be (5, T)
print(mod.nobs)             # should be T
