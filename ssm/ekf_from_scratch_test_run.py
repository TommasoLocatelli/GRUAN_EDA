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


# test filter run
if False:
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

# test smoother run
if False:
    smooth_s_hist, smooth_p_hist, smooth_gains_hist, lag_one_cov_hist = kf.smooth()
    smoothed_obs = array([A(smooth_s_hist[i]) for i in range(len(smooth_s_hist))])
    smoothed_states_var = array([np.diag(smooth_p_hist[i]) for i in range(len(smooth_p_hist))])
    smoothed_states_unc = np.sqrt(smoothed_states_var)

    thv = smooth_s_hist[:, Thv_S].reshape(-1)
    p   = smooth_s_hist[:, P_S].reshape(-1)
    r   = smooth_s_hist[:, R_S].reshape(-1)

    thv_uc = smoothed_states_unc[:, Thv_S].reshape(-1)
    p_uc   = smoothed_states_unc[:, P_S].reshape(-1)
    r_uc   = smoothed_states_unc[:, R_S].reshape(-1)
    smoothed_temp_unc = gp.virtual_potential_temperature_inverse_uncertainty(
        thv, p, r,thv_uc, p_uc, r_uc).reshape(-1)

    smoothed_measurement_unc = array([
        [smoothed_states_unc[i][Z_S], smoothed_states_unc[i][LZ_S], smoothed_temp_unc[i], smoothed_states_unc[i][P_S],
         smoothed_states_unc[i][RH_S], smoothed_states_unc[i][R_S], smoothed_states_unc[i][U_S], smoothed_states_unc[i][V_S]]
        for i in range(len(smoothed_states_unc))
    ])

    smoothed_measurement_unc *= 2

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
        ax.plot(time, smoothed_obs[:, i], label="Smoothed", color="green", alpha=0.6)
        ax.fill_between(time, smoothed_obs[:, i].flatten() - np.array(smoothed_measurement_unc)[:, i].flatten(),
                        smoothed_obs[:, i].flatten() + np.array(smoothed_measurement_unc)[:, i].flatten(), color='lightgreen', alpha=0.2, label='Smoothing Uncertainty')
        ax.set_title(f"Component {vars[i]}")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# test EM algorithm run
if False:
    EM_log_likelihoods, EM_Q, EM_s_0, EM_p_0=kf.EM_algorithm(max_iter=100, tol=1, verbose=True)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(EM_log_likelihoods) + 1), EM_log_likelihoods, linewidth=1)
    plt.title('EM log-likelihood vs iteration')
    plt.xlabel('EM iteration')
    plt.ylabel('Log-likelihood')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    kf = ExtendedKalmanFilter(Phi, EM_Q[-1], A, J_A, EM_s_0[-1], EM_p_0[-1], obs, meas_var)
    smooth_s_hist, smooth_p_hist, smooth_gains_hist, lag_one_cov_hist = kf.smooth()
    
    smoothed_obs = array([A(smooth_s_hist[i]) for i in range(len(smooth_s_hist))])
    smoothed_states_var = array([np.diag(smooth_p_hist[i]) for i in range(len(smooth_p_hist))])
    smoothed_states_unc = np.sqrt(smoothed_states_var)

    thv = smooth_s_hist[:, Thv_S].reshape(-1)
    p   = smooth_s_hist[:, P_S].reshape(-1)
    r   = smooth_s_hist[:, R_S].reshape(-1)

    thv_uc = smoothed_states_unc[:, Thv_S].reshape(-1)
    p_uc   = smoothed_states_unc[:, P_S].reshape(-1)
    r_uc   = smoothed_states_unc[:, R_S].reshape(-1)
    smoothed_temp_unc = gp.virtual_potential_temperature_inverse_uncertainty(
        thv, p, r,thv_uc, p_uc, r_uc).reshape(-1)

    smoothed_measurement_unc = array([
        [smoothed_states_unc[i][Z_S], smoothed_states_unc[i][LZ_S], smoothed_temp_unc[i], smoothed_states_unc[i][P_S],
         smoothed_states_unc[i][RH_S], smoothed_states_unc[i][R_S], smoothed_states_unc[i][U_S], smoothed_states_unc[i][V_S]]
        for i in range(len(smoothed_states_unc))
    ])

    smoothed_measurement_unc *= 2

    # convert obs to array (n, q)
    obs_arr = np.array(obs).reshape(len(obs), -1)

    if scales_params is not None:
        z_params = scales_params.get("z", {})
        other_params = scales_params.get("other", {})
        obs_arr[:, Z_I] = obs_arr[:, Z_I] * z_params.get("range", 1) + z_params.get("min", 0)
        obs_arr[:, LZ_I] = obs_arr[:, LZ_I] * z_params.get("range", 1)
        obs_arr[:, [T_I, P_I, RH_I, R_I, U_I, V_I]] = obs_arr[:, [T_I, P_I, RH_I, R_I, U_I, V_I]] * other_params.get("range", 1) + other_params.get("min", 0)
        smoothed_obs[:, Z_I] = smoothed_obs[:, Z_I] * z_params.get("range", 1) + z_params.get("min", 0)
        smoothed_obs[:, LZ_I] = smoothed_obs[:, LZ_I] * z_params.get("range", 1)
        smoothed_obs[:, [T_I, P_I, RH_I, R_I, U_I, V_I]] = smoothed_obs[:, [T_I, P_I, RH_I, R_I, U_I, V_I]] * other_params.get("range", 1) + other_params.get("min", 0)
        smoothed_measurement_unc[:, Z_I] = smoothed_measurement_unc[:, Z_I] * z_params.get("range", 1)
        smoothed_measurement_unc[:, LZ_I] = smoothed_measurement_unc[:, LZ_I] * z_params.get("range", 1)
        smoothed_measurement_unc[:, [T_I, P_I, RH_I, R_I, U_I, V_I]] = smoothed_measurement_unc[:, [T_I, P_I, RH_I, R_I, U_I, V_I]] * other_params.get("range", 1)

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
        ax.plot(time, smoothed_obs[:, i], label="Smoothed", color="green", alpha=0.6)
        ax.fill_between(time, smoothed_obs[:, i].flatten() - np.array(smoothed_measurement_unc)[:, i].flatten(),
                        smoothed_obs[:, i].flatten() + np.array(smoothed_measurement_unc)[:, i].flatten(), color='lightgreen', alpha=0.2, label='Smoothing Uncertainty')
        ax.set_title(f"Component {vars[i]}")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()

    # how to denormalized theta v?
    # 1. find the timestamp with minimum and maximum normalized theta v
    # 2. apply the poisson equation to the denormalized temperature, pressur and r at this timestamps
    #    and find the equivalent normalization parameters of theta v
    # still to be implemented

    n_panels = 4
    cols = 2
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = axes.flatten()

    vars=['Thv', 'LThv', 'LRH', 'Lu / Lv']

    state_indices = [Thv_S, LThv_S, LRH_S, (LU_S, LV_S)]

    for i in range(n_panels):
        ax = axes[i]
        if i < 3:
            idx = state_indices[i]
            state_series = smooth_s_hist[:, idx]
            state_unc = smoothed_states_unc[:, idx]
            ax.plot(time, state_series, label="Smoothed State", color="green", alpha=0.6)
            ax.fill_between(time,
                            state_series.flatten() - state_unc.flatten() * 2,
                            state_series.flatten() + state_unc.flatten() * 2,
                            color='lightgreen', alpha=0.2, label='Smoothed State Uncertainty')
        else:
            idx_u, idx_v = state_indices[i]
            state_series_u = smooth_s_hist[:, idx_u]
            state_unc_u = smoothed_states_unc[:, idx_u]
            state_series_v = smooth_s_hist[:, idx_v]
            state_unc_v = smoothed_states_unc[:, idx_v]

            ax.plot(time, state_series_u, label="Smoothed Lu", color="green", alpha=0.6)
            ax.fill_between(time,
                            state_series_u.flatten() - state_unc_u.flatten() * 2,
                            state_series_u.flatten() + state_unc_u.flatten() * 2,
                            color='lightgreen', alpha=0.2)
            ax.plot(time, state_series_v, label="Smoothed Lv", color="darkgreen", alpha=0.6)
            ax.fill_between(time,
                            state_series_v.flatten() - state_unc_v.flatten() * 2,
                            state_series_v.flatten() + state_unc_v.flatten() * 2,
                            color='palegreen', alpha=0.2)
        ax.set_title(f"State Component {vars[i]}")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()

# test simulazioni
if True:
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
