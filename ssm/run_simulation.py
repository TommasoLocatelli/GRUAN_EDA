import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ssm.preproc import preprocess_profile
from ssm.ssm_model import PHI, A, J_A
from ssm.ssm_model import (
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I,
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S
)
from ssm.guess_starting_values import guess_initial_state
from ssm.ekf import ExtendedKalmanFilter
from ssm.em_ekf import EKF_EM
import gruanpy as gp
from ssm.standardize import standardize_obs, denormalize_obs, reconstruct_physical_states


example_paths = [
    r"gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc",
    r"gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc",
    r"gdp\products_RS41-GDP-1_PAY_2024\PAY-RS-01_2_RS41-GDP_001_20240109T120000_1-002-001.nc",
    r"gdp\products_RS41-GDP-1_TEN_2024\TEN-RS-01_2_RS41-GDP_001_20240103T110000_1-000-001.nc",
]

path = example_paths[0]

# ---------------------------------------------------------
# 1. preprocess: obs, meas_var (PHYSICAL)
# ---------------------------------------------------------
obs, meas_var = preprocess_profile(path, upper_bound=3000)

obs, meas_var, _ = standardize_obs(obs, meas_var)
meas_unc_95 = 2 * np.sqrt(meas_var)

# ---------------------------------------------------------
# 2. initial state and covariance (PHYSICAL)
# ---------------------------------------------------------
s0, P0 = guess_initial_state(obs, meas_var)

# ---------------------------------------------------------
# 3. initial process noise (PHYSICAL)
# ---------------------------------------------------------
Q_scale = 1e3
Q0 = np.eye(len(s0)) * Q_scale

# ---------------------------------------------------------
# 4. EM on Q, s0, P0
# ---------------------------------------------------------
state_min = np.full(12, -np.inf)
state_max = np.full(12,  np.inf)

state_min[P_S] = 1e-3
state_min[R_S] = 1e-8
state_min[RH_S] = 0.0
state_max[RH_S] = 120.0

kf = ExtendedKalmanFilter(
    PHI, Q0, A, J_A, s0, P0, obs, meas_var,
    state_min=state_min,
    state_max=state_max,
)

em = EKF_EM(kf)

EM_log_likelihoods, EM_Q, EM_s0, EM_P0 = em.EM_algorithm(
    max_iter=100,
    tol=1e-3,
    verbose=True
)

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(EM_log_likelihoods) + 1), EM_log_likelihoods)
plt.title("EM log-likelihood vs iteration")
plt.xlabel("EM iteration")
plt.ylabel("Log-likelihood")
plt.grid(True)
plt.tight_layout()
plt.show()

Q_em  = EM_Q[-1]
s0_em = EM_s0[-1]
P0_em = EM_P0[-1]

# ---------------------------------------------------------
# 5. EKF + RTS smoother with EM-estimated parameters
# ---------------------------------------------------------
kf_em = ExtendedKalmanFilter(
    PHI, Q_em, A, J_A, s0_em, P0_em, obs, meas_var,
    state_min=state_min,
    state_max=state_max,
)

kf_em.filter()
smooth_s_hist, smooth_p_hist, smooth_gains_hist, lag_one_cov_hist = kf_em.smooth()

# ---------------------------------------------------------
# 6. simulate from EM-estimated model
# ---------------------------------------------------------
sim_states = kf_em.simulate_states()          # shape (n, 12)
sim_obs = kf_em.simulate_observations(sim_states)  # shape (n, 8)

time = np.arange(obs.shape[0])

# ---------------------------------------------------------
# 7. plot SMOOTHED STATE vs SIMULATED STATE
# ---------------------------------------------------------
state_names = ["z", "Lz", "θv", "Lθv", "p", "RH", "LRH", "r", "u", "Lu", "v", "Lv"]

fig, axes = plt.subplots(3, 4, figsize=(20, 10))
axes = axes.flatten()

for i in range(12):
    ax = axes[i]
    ax.plot(time, smooth_s_hist[:, i], label="Smoothed", color="orange", alpha=0.8)
    ax.plot(time, sim_states[:, i], label="Simulated", color="blue", alpha=0.6)
    ax.set_title(state_names[i])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. plot OBSERVED vs SIMULATED OBS
# ---------------------------------------------------------
obs_names = ["z", "Lz", "T", "p", "RH", "r", "u", "v"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i in range(8):
    ax = axes[i]
    ax.plot(time, obs[:, i], label="Observed", color="blue", alpha=0.7)
    ax.plot(time, sim_obs[:, i], label="Simulated", color="orange", alpha=0.7)
    ax.set_title(obs_names[i])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
