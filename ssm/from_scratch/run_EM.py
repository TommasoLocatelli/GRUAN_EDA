import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ssm.from_scratch.preproc import preprocess_profile
from ssm.from_scratch.ssm_model import PHI, A, J_A
from ssm.from_scratch.ssm_model import (
    Z_I, LZ_I, T_I, P_I, RH_I, R_I, U_I, V_I,
    Z_S, LZ_S, Thv_S, LThv_S, P_S, RH_S, LRH_S, R_S, U_S, LU_S, V_S, LV_S
)
from ssm.from_scratch.guess_starting_values import guess_initial_state
from ssm.from_scratch.ekf import ExtendedKalmanFilter
from ssm.from_scratch.em_ekf import EKF_EM   # <-- import your EM wrapper
import gruanpy as gp

from ssm.from_scratch.standardize import standardize_obs, denormalize_obs, reconstruct_physical_states


example_paths = [
    r"gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc",
    r"gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc",
    r"gdp\products_RS41-GDP-1_PAY_2024\PAY-RS-01_2_RS41-GDP_001_20240109T120000_1-002-001.nc",
    r"gdp\products_RS41-GDP-1_TEN_2024\TEN-RS-01_2_RS41-GDP_001_20240103T110000_1-000-001.nc",
]

path = example_paths[0]

# ---------------------------------------------------------
# 1. preprocess: obs, meas_var (PHYSICAL, n,8)
# ---------------------------------------------------------
obs, meas_var = preprocess_profile(path, upper_bound=3000)

#obs, meas_var, _ = standardize_obs(obs, meas_var)

# measurement uncertainty (95% CI)
meas_unc_95 = 2 * np.sqrt(meas_var)

# ---------------------------------------------------------
# 2. initial state and covariance (PHYSICAL)
# ---------------------------------------------------------
s0, P0 = guess_initial_state(obs, meas_var)

# ---------------------------------------------------------
# 3. initial process noise (PHYSICAL)
# ---------------------------------------------------------
Q_scale = 1e1
Q0 = np.eye(len(s0)) * Q_scale
Q0[LThv_S, LThv_S] *= 1e3 ### LThv sempre positiva?!

# ---------------------------------------------------------
# 4. EM algorithm on Q, s0, P0
# ---------------------------------------------------------
state_min = np.full(12, -1e12)
state_max = np.full(12,  1e12)

state_min[P_S] = 1e-3      # p > 0
state_min[R_S] = 1e-8      # r > 0
state_min[RH_S] = 0.0      # RH >= 0 (in %)
state_max[RH_S] = 120.0    # RH <= 120 (in %)

kf = ExtendedKalmanFilter(
    PHI, Q0, A, J_A, s0, P0, obs, meas_var,
    state_min=state_min,
    state_max=state_max,
)

em = EKF_EM(kf)   # <-- wrap the EKF

EM_log_likelihoods, EM_Q, EM_s0, EM_P0 = em.EM_algorithm(
    max_iter=100,
    tol=1e-3,
    verbose=True
)

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(EM_log_likelihoods) + 1), EM_log_likelihoods, linewidth=1)
plt.title("EM log-likelihood vs iteration")
plt.xlabel("EM iteration")
plt.ylabel("Log-likelihood")
plt.grid(True)
plt.tight_layout()
plt.show()

# prendi gli ultimi parametri stimati
Q_em  = EM_Q[-1]
s0_em = EM_s0[-1]
P0_em = EM_P0[-1]

# ---------------------------------------------------------
# 5. EKF + RTS smoother with EM-estimated parameters
# ---------------------------------------------------------
kf = ExtendedKalmanFilter(
    PHI, Q_em, A, J_A, s0_em, P0_em, obs, meas_var,
    state_min=state_min,
    state_max=state_max,
)

kf.filter()
smooth_s_hist, smooth_p_hist, smooth_gains_hist, lag_one_cov_hist = kf.smooth()

# ---------------------------------------------------------
# 6. map smoothed states to observation space (PHYSICAL)
# ---------------------------------------------------------
smoothed_obs = np.vstack([A(smooth_s_hist[t]) for t in range(len(smooth_s_hist))])

smoothed_states_var = np.array([np.diag(smooth_p_hist[t]) for t in range(len(smooth_p_hist))])
smoothed_states_unc = np.sqrt(smoothed_states_var)

# ---------------------------------------------------------
# 7. propagate θv uncertainty to T uncertainty
# ---------------------------------------------------------
thv = smooth_s_hist[:, Thv_S]
p_phys = smooth_s_hist[:, P_S]
r_phys = smooth_s_hist[:, R_S]

thv_uc = smoothed_states_unc[:, Thv_S]
p_uc = smoothed_states_unc[:, P_S]
r_uc = smoothed_states_unc[:, R_S]

smoothed_temp_unc = gp.virtual_potential_temperature_inverse_uncertainty(
    thv, p_phys, r_phys, thv_uc, p_uc, r_uc
)

smoothed_measurement_unc = np.column_stack([
    2 * smoothed_states_unc[:, Z_S],
    2 * smoothed_states_unc[:, LZ_S],
    2 * smoothed_temp_unc,
    2 * smoothed_states_unc[:, P_S],
    2 * smoothed_states_unc[:, RH_S],
    2 * smoothed_states_unc[:, R_S],
    2 * smoothed_states_unc[:, U_S],
    2 * smoothed_states_unc[:, V_S],
])

# ---------------------------------------------------------
# 8. plotting: OBSERVATIONS vs EM-SMOOTHED
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

    ax.plot(time, smoothed_obs[:, i], label="EM-smoothed", color="green", alpha=0.6)
    ax.fill_between(
        time,
        smoothed_obs[:, i] - smoothed_measurement_unc[:, i],
        smoothed_obs[:, i] + smoothed_measurement_unc[:, i],
        color="lightgreen",
        alpha=0.2,
        label="Smoothing 95% CI",
    )

    ax.set_title(f"Component {vars[i]}")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. plotting: SMOOTHED STATE COMPONENTS (12 variables)
# ---------------------------------------------------------
state_vars = ["z", "Lz", "θv", "Lθv", "p", "RH", "LRH", "r", "u", "Lu", "v", "Lv"]

fig2, axes2 = plt.subplots(3, 4, figsize=(20, 10))
axes2 = axes2.flatten()

state_unc_95 = 2 * smoothed_states_unc

for i in range(12):
    ax = axes2[i]

    ax.plot(time, smooth_s_hist[:, i], color="orange", alpha=0.8, label="Smoothed state")

    ax.fill_between(
        time,
        smooth_s_hist[:, i] - state_unc_95[:, i],
        smooth_s_hist[:, i] + state_unc_95[:, i],
        color="salmon",
        alpha=0.2,
        label="95% CI"
    )

    ax.set_title(state_vars[i])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
