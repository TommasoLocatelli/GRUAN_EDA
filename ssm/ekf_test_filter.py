import numpy as np
from numpy import eye, array, asarray
import sys
import os
import matplotlib.pyplot as plt
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
from ssm.ekf_from_scratch import ExtendedKalmanFilter
from ssm.ekf_prep import prep_ekf

example_paths = [
    r'gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc',
    r'gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc',
    r'gdp\products_RS41-GDP-1_PAY_2024\PAY-RS-01_2_RS41-GDP_001_20240109T120000_1-002-001.nc',
    r'gdp\products_RS41-GDP-1_TEN_2024\TEN-RS-01_2_RS41-GDP_001_20240103T110000_1-000-001.nc'
]

obs, meas_var, Phi, Q, A, J_A, s_0, P_0 , scales_params= prep_ekf(example_paths[0], upper_bound=3000, 
                                                   Q_scale=10000, 
                                                   normalize_method='minmax')

meas_unc = np.sqrt(meas_var)*2

kf = ExtendedKalmanFilter(Phi, Q, A, J_A, s_0, P_0, obs, meas_var)

# obs vector
# x_t = [z_t, Lz_t, T_t, p_t, RH_t, r_t, u_t, v_t]
# obs indexes
Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I = range(8)

# state vector
# s_t = [z_t, Lz_t, Thv_t, LThv_t, p_t, RH_t, LRH_t, r_t, u_t, Lu_t, v_t, Lv_t]
# state indexes
Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S = range(12)


# test filter run

s_pred_hist, p_pred_hist, s_upd_hist, p_upd_hist, gains_hist = kf.filter()
filtered_obs = array([A(s_upd_hist[i]) for i in range(len(s_upd_hist))])
filtered_states_var = array([np.diag(p_upd_hist[i]) for i in range(len(p_upd_hist))])
filtered_states_unc = np.sqrt(filtered_states_var)

thv = s_upd_hist[:, Thv_S].reshape(-1)
p   = s_upd_hist[:, P_S].reshape(-1)
r   = s_upd_hist[:, R_S].reshape(-1)

# filtered_states_unc already contains 1-sigma standard deviations for state components
thv_uc = filtered_states_unc[:, Thv_S].reshape(-1)
p_uc   = filtered_states_unc[:, P_S].reshape(-1)
r_uc   = filtered_states_unc[:, R_S].reshape(-1)

filtered_temp_unc = gp.virtual_potential_temperature_inverse_uncertainty(
    thv, p, r, thv_uc, p_uc, r_uc).reshape(-1)

filtered_measurement_unc = array([
    [filtered_states_unc[i][Z_S], filtered_states_unc[i][LZ_S], filtered_temp_unc[i], filtered_states_unc[i][P_S],
        filtered_states_unc[i][RH_S], filtered_states_unc[i][R_S], filtered_states_unc[i][U_S], filtered_states_unc[i][V_S]]
    for i in range(len(filtered_states_var))
])*2 # multiply by 2 to get ~95% confidence intervals

# convert obs to array (n, q)
obs_arr = np.array(obs).reshape(len(obs), -1)

# number of panels
n_panels = 8
cols = 4
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(18, 8))
axes = axes.flatten()

time = np.arange(len(obs))

vars=['z', 'Lz', 'T', 'p', 'RH', 'r', 'u', 'v']
for i in range(n_panels):
    ax = axes[i]
    ax.plot(time, obs_arr[:, i], label="Observation", color="blue", alpha=0.6)
    ax.fill_between(time, obs_arr[:, i].flatten() - np.array(meas_unc)[:, i].flatten(),
                    obs_arr[:, i].flatten() + np.array(meas_unc)[:, i].flatten(), color='lightblue', alpha=0.2, label='Measurement Uncertainty')
    ax.plot(time, filtered_obs[:, i], label="Filtered", color="orange", alpha=0.6)
    ax.fill_between(time, filtered_obs[:, i].flatten() - filtered_measurement_unc[:, i].flatten(),
                    filtered_obs[:, i].flatten() + filtered_measurement_unc[:, i].flatten(), color='salmon', alpha=0.2, label='Filter Uncertainty')
    ax.set_title(f"Component {vars[i]}")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
