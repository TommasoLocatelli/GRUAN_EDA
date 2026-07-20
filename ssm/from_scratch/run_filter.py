import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ssm.from_scratch.preproc import preprocess_profile
from ssm.from_scratch.ssm_model import PHI, A, J_A
from ssm.from_scratch.ssm_model import (
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I,
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S
)
from ssm.from_scratch.guess_starting_values import guess_initial_state
from ssm.from_scratch.ekf import ExtendedKalmanFilter
import gruanpy as gp
from ssm.from_scratch.standardize import standardize_obs, denormalize_obs, reconstruct_physical_states


example_paths = [
    r"gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc",
    r"gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc",
    r"gdp\products_RS41-GDP-1_PAY_2024\PAY-RS-01_2_RS41-GDP_001_20240109T120000_1-002-001.nc",
    r"gdp\products_RS41-GDP-1_TEN_2024\TEN-RS-01_2_RS41-GDP_001_20240103T110000_1-000-001.nc",
]

path = example_paths[1]

# ---------------------------------------------------------
# 1. preprocess: obs, meas_var (physical, n,8)
# ---------------------------------------------------------
obs, meas_var = preprocess_profile(path, upper_bound=3000)

#obs, meas_var, _ = standardize_obs(obs, meas_var)

# measurement uncertainty (1σ) and 95% CI
meas_unc = np.sqrt(meas_var)
meas_unc_95 = 2 * meas_unc

# ---------------------------------------------------------
# 2. guess initial state and covariance (physical)
# ---------------------------------------------------------
s0, P0 = guess_initial_state(obs, meas_var)

# ---------------------------------------------------------
# 3. define process noise (physical)
# ---------------------------------------------------------
Q_scale = 1e3
Q = np.eye(len(s0)) * Q_scale

# ---------------------------------------------------------
# 4. run EKF in physical space
# ---------------------------------------------------------
state_min = np.full(12, -np.inf)
state_max = np.full(12,  np.inf)

state_min[P_S] = 1e-3      # p > 0
state_min[R_S] = 1e-8      # r > 0
state_min[RH_S] = 0.0      # RH >= 0
state_max[RH_S] = 120.0      # RH <= 120

kf = ExtendedKalmanFilter(
    PHI, Q, A, J_A, s0, P0, obs, meas_var,
    state_min=state_min,
    state_max=state_max,
)

s_pred_hist, p_pred_hist, s_upd_hist, p_upd_hist, gains_hist = kf.filter()

# ---------------------------------------------------------
# 5. map filtered states to observation space (physical)
# ---------------------------------------------------------
filtered_obs = np.vstack([A(s_upd_hist[t]) for t in range(len(s_upd_hist))])

# state variances and uncertainties
filtered_states_var = np.array([np.diag(p_upd_hist[t]) for t in range(len(p_upd_hist))])
filtered_states_unc = np.sqrt(filtered_states_var)

# ---------------------------------------------------------
# 6. propagate θv uncertainty to T uncertainty
# ---------------------------------------------------------
thv = s_upd_hist[:, Thv_S]
p_phys = s_upd_hist[:, P_S]
r_phys = s_upd_hist[:, R_S]

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
# 7. plotting (physical)
# ---------------------------------------------------------
n_panels = 8
cols = 4
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(18, 8))
axes = axes.flatten()

time = np.arange(obs.shape[0])
vars = ["z", "Lz", "T", "p", "RH", "r", "u", "v"]

for i in range(n_panels):
    ax = axes[i]

    ax.plot(time, obs[:, i], label="Observation", color="blue", alpha=0.6)
    ax.fill_between(
        time,
        obs[:, i] - meas_unc_95[:, i],
        obs[:, i] + meas_unc_95[:, i],
        color="lightblue",
        alpha=0.2,
        label="Measurement 95% CI",
    )

    ax.plot(time, filtered_obs[:, i], label="Filtered", color="orange", alpha=0.6)
    ax.fill_between(
        time,
        filtered_obs[:, i] - filtered_measurement_unc[:, i],
        filtered_obs[:, i] + filtered_measurement_unc[:, i],
        color="salmon",
        alpha=0.2,
        label="Filter 95% CI",
    )

    ax.set_title(f"Component {vars[i]}")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. Plot delle 12 componenti dello stato filtrato
# ---------------------------------------------------------

state_vars = ["z", "Lz", "θv", "Lθv", "p", "RH", "LRH", "r", "u", "Lu", "v", "Lv"]

fig2, axes2 = plt.subplots(3, 4, figsize=(20, 10))
axes2 = axes2.flatten()

time = np.arange(s_upd_hist.shape[0])

# estrai incertezze (1σ → 95% CI = 2σ)
state_unc_95 = 2 * filtered_states_unc

for i in range(12):
    ax = axes2[i]

    # valore filtrato
    ax.plot(time, s_upd_hist[:, i], color="orange", alpha=0.8, label="Filtered state")

    # banda di incertezza 95%
    ax.fill_between(
        time,
        s_upd_hist[:, i] - state_unc_95[:, i],
        s_upd_hist[:, i] + state_unc_95[:, i],
        color="salmon",
        alpha=0.2,
        label="95% CI"
    )

    ax.set_title(state_vars[i])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
