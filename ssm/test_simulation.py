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
                                                   Q_scale=1000, 
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

# test simulazioni

EM_log_likelihoods, EM_Q, EM_s_0, EM_p_0=kf.EM_algorithm(max_iter=100, tol=1, verbose=True)
kf = ExtendedKalmanFilter(Phi, EM_Q[-1], A, J_A, EM_s_0[-1], EM_p_0[-1], obs, meas_var)
smooth_s, smooth_p, _, _ = kf.smooth()
sim_states = kf.simulate_states()
sim_obs = kf.simulate_observations(sim_states)

# --- PLOT SMOOTHED STATE VS SIMULATED STATE ---

smooth_s_hist = smooth_s.reshape(kf.n, kf.p)
sim_states_hist = sim_states.reshape(kf.n, kf.p)

time = np.arange(kf.n)

state_names = ['z', 'Lz', 'Thv', 'LThv', 'p', 'RH', 'LRH', 'r', 'u', 'Lu', 'v', 'Lv']

n_panels = kf.p
cols = 4
rows = int(np.ceil(n_panels / cols))

fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
axes = axes.flatten()

for i in range(n_panels):
    ax = axes[i]
    ax.plot(time, smooth_s_hist[:, i], label="Smoothed", color="orange", alpha=0.8)
    ax.plot(time, sim_states_hist[:, i], label="Simulated", color="blue", alpha=0.6)
    ax.set_title(state_names[i])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# --- PLOT OBSERVED VS SIMULATED OBSERVATIONS ---

obs_arr = np.array(obs).reshape(kf.n, kf.q)
sim_obs_arr = sim_obs.reshape(kf.n, kf.q)

time = np.arange(kf.n)

obs_names = ['z', 'Lz', 'T', 'p', 'RH', 'r', 'u', 'v']

n_panels = kf.q
cols = 4
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(18, 8))
axes = axes.flatten()

for i in range(n_panels):
    ax = axes[i]
    ax.plot(time, obs_arr[:, i], label="Observed", color="blue", alpha=0.7)
    ax.plot(time, sim_obs_arr[:, i], label="Simulated", color="orange", alpha=0.7)
    ax.set_title(obs_names[i])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
