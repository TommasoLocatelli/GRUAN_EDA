import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ssm.preproc import preprocess_profile
from ssm.standardize import standardize_obs, denormalize_obs, denormalize_states
from ssm.ssm_model import PHI, A, J_A
from ssm.ssm_model import (
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I,
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S
)
from ssm.guess_starting_values import guess_initial_state
from ssm.ekf import ExtendedKalmanFilter
import gruanpy as gp


example_paths = [
    r"gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc",
    r"gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc",
    r"gdp\products_RS41-GDP-1_PAY_2024\PAY-RS-01_2_RS41-GDP_001_20240109T120000_1-002-001.nc",
    r"gdp\products_RS41-GDP-1_TEN_2024\TEN-RS-01_2_RS41-GDP_001_20240103T110000_1-000-001.nc",
]

path = example_paths[0]

# ---------------------------------------------------------
# 1. preprocess: obs, meas_var (n,8)
# ---------------------------------------------------------
obs, meas_var = preprocess_profile(path, upper_bound=3000)

# measurement uncertainty (1σ) and 95% CI
meas_unc = np.sqrt(meas_var)
meas_unc_95 = 2 * meas_unc

# ---------------------------------------------------------
# 2. standardize obs + meas_var (min–max)
# ---------------------------------------------------------
obs_std, meas_var_std, params = standardize_obs(obs, meas_var)

# ---------------------------------------------------------
# 3. guess initial state and covariance
# ---------------------------------------------------------
s0, P0 = guess_initial_state(obs_std, meas_var_std)

# ---------------------------------------------------------
# 4. define process noise
# ---------------------------------------------------------
Q_scale = 1e4
Q = np.eye(len(s0)) * Q_scale

# ---------------------------------------------------------
# 5. run EKF
# ---------------------------------------------------------
state_min = np.full(12, -np.inf)
state_max = np.full(12,  np.inf)

# vincoli fisici
state_min[P_S] = 1e-3      # pressione > 0
state_min[R_S] = 1e-8      # r > 0
state_min[RH_S] = 0.0      # RH >= 0
state_max[RH_S] = 1.0      # RH <= 1

kf = ExtendedKalmanFilter(
    PHI, Q, A, J_A, s0, P0, obs_std, meas_var_std,
    state_min=state_min,
    state_max=state_max,
)

s_pred_hist, p_pred_hist, s_upd_hist, p_upd_hist, gains_hist = kf.filter()

# ---------------------------------------------------------
# 6. map filtered states to observation space
# ---------------------------------------------------------
filtered_obs_std = np.vstack([A(s_upd_hist[t]) for t in range(len(s_upd_hist))])

# state variances and uncertainties
filtered_states_var = np.array([np.diag(p_upd_hist[t]) for t in range(len(p_upd_hist))])
filtered_states_unc = np.sqrt(filtered_states_var)

# ---------------------------------------------------------
# 7. denormalize obs and states
# ---------------------------------------------------------
obs_phys = denormalize_obs(obs_std, params)
filtered_obs_phys = denormalize_obs(filtered_obs_std, params)
states_phys = denormalize_states(s_upd_hist, obs_phys)

# ---------------------------------------------------------
# 8. propagate θv uncertainty to T uncertainty
# ---------------------------------------------------------
thv = states_phys[:, Thv_S]
p_phys = states_phys[:, P_S]
r_phys = states_phys[:, R_S]

thv_uc = filtered_states_unc[:, Thv_S]
p_uc = filtered_states_unc[:, P_S]
r_uc = filtered_states_unc[:, R_S]

filtered_temp_unc = gp.virtual_potential_temperature_inverse_uncertainty(
    thv, p_phys, r_phys, thv_uc, p_uc, r_uc
)

# build measurement uncertainty from state uncertainties (95% CI)
filtered_measurement_unc = np.column_stack([
    2 * filtered_states_unc[:, Z_S],
    2 * filtered_states_unc[:, LZ_S],
    2 * filtered_temp_unc,
    2 * filtered_states_unc[:, P_S],
    2 * filtered_states_unc[:, RH_S],
    2 * filtered_states_unc[:, R_S],
    2 * filtered_states_unc[:, U_S],
    2 * filtered_states_unc[:, V_S],
])

# ---------------------------------------------------------
# 9. plotting
# ---------------------------------------------------------
n_panels = 8
cols = 4
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(18, 8))
axes = axes.flatten()

time = np.arange(obs_phys.shape[0])
vars = ["z", "Lz", "T", "p", "RH", "r", "u", "v"]

for i in range(n_panels):
    ax = axes[i]

    ax.plot(time, obs_phys[:, i], label="Observation", color="blue", alpha=0.6)
    ax.fill_between(
        time,
        obs_phys[:, i] - meas_unc_95[:, i],
        obs_phys[:, i] + meas_unc_95[:, i],
        color="lightblue",
        alpha=0.2,
        label="Measurement 95% CI",
    )

    ax.plot(time, filtered_obs_phys[:, i], label="Filtered", color="orange", alpha=0.6)
    ax.fill_between(
        time,
        filtered_obs_phys[:, i] - filtered_measurement_unc[:, i],
        filtered_obs_phys[:, i] + filtered_measurement_unc[:, i],
        color="salmon",
        alpha=0.2,
        label="Filter 95% CI",
    )

    ax.set_title(f"Component {vars[i]}")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
