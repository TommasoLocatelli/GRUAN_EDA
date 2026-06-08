import numpy as np
from numpy import eye, array, asarray
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
from ssm.ekf_from_scratch import ExtendedKalmanFilter
from ssm.ekf_from_scratch_test_prep import prep_ekf

example_paths = [
    r'gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc',
    r'gdp\icm16\POT-RS-01_2_RS41-GDP_001_20250319T135500_1-000-001.nc',
    r'gdp\products_RS41-GDP-1_PAY_2024\PAY-RS-01_2_RS41-GDP_001_20240109T120000_1-002-001.nc',
    r'gdp\products_RS41-GDP-1_TEN_2024\TEN-RS-01_2_RS41-GDP_001_20240103T110000_1-000-001.nc'
]

obs, meas_var, Phi, Q, A, J_A, s_0, P_0 = prep_ekf(example_paths[0], upper_bound=3000)
kf = ExtendedKalmanFilter(Phi, Q, A, J_A, s_0, P_0, obs, meas_var)
    
# test filter run
if True:
    s_pred_hist, p_pred_hist, s_upd_hist, p_upd_hist, gains_hist = kf.filter()
    # convert obs to array (n, q)
    obs_arr = np.array(obs).reshape(len(obs), -1)

    # number of panels
    n_panels = 8
    cols = 4
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(18, 8))
    axes = axes.flatten()

    time = np.arange(len(obs))

    for i in range(n_panels):

        ax = axes[i]

        # plot observation (if dimension matches)
        if i < obs_arr.shape[1]:
            ax.plot(time, obs_arr[:, i], label="Observation", color="black", alpha=0.6)

        # plot filtered state component
        if i < s_upd_hist.shape[1]:
            ax.plot(time, s_upd_hist[:, i, 0], label="Filtered", color="tab:blue")

        ax.set_title(f"Component {i}")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


raise NotImplementedError("This test run script is not yet implemented. The code is still being developed and debugged. Please check back later for the complete implementation.")
kf = ExtendedKalmanFilter(Phi, Q, A, J_A, s_0, P_0, obs, meas_var)

EM_log_likelihoods, EM_Q, EM_s_0, EM_p_0=kf.EM_algorithm(max_iter=1000, tol=1, verbose=True)
kf = ExtendedKalmanFilter(Phi, EM_Q[-1], A, J_A, EM_s_0[-1], EM_p_0[-1], obs, meas_var)

state_pred, prec_prd, state_estimates, precision_estimates, gains = kf.filter()
state_estimates, precision_estimates, smoothing_gains = kf.smooth()
smoothed_obs= array([A(state_estimates[i]) for i in range(len(state_estimates))])
smoothed_var =[np.diag(precision_estimates[i]) for i in range(len(precision_estimates))]

# NEED TO COMPUTE T unc
theta_sm_var=smoothed_var[Thv_S]
p_sm_var=smoothed_var[P_S]
r_sm_Var=smoothed_var[R_S]
#print(smoothed_var, theta_sm_var.shape)

def T_var(thv, thv_var, 
        p, p_var,
        r, r_var):
    factor_p = (p / p0) ** kappa
    factor_r = 1.0 / (1.0 + 0.61 * r)
    dT_dthv = factor_p * factor_r
    dT_dp   = thv * kappa * (p ** (kappa - 1)) / (p0 ** kappa) * factor_r
    dT_dr   = -0.61 * thv * factor_p * (factor_r ** 2)

    T_var=(dT_dthv**2)*thv_var+(dT_dp**2)*p_var+(dT_dr**2)*r_var
    return T_var
#smoothed_unc = np.sqrt(smoothed_var)*2

plt.plot(EM_log_likelihoods)
plt.title('EM -loglik per iteration')
plt.show()

for i, var in enumerate(['z', 'Lz', 'T', 'p', 'RH', 'r', 'u', 'v']):
    plt.figure()
    plt.plot(smoothed_obs[:, i], label='Kalman Filter Estimate', color='orange')
    #plt.fill_between(range(len(smoothed_obs)),
    #                 smoothed_obs[:, i].flatten() - smoothed_unc[:, i].flatten(),
    #                 smoothed_obs[:, i].flatten() + smoothed_unc[:, i].flatten(),
    #                 color='#EB8D35', alpha=0.2, label='Filter Uncertainty')
    plt.plot(np.array(obs)[:, i], label='Observations', alpha=0.5, color='blue')
    plt.fill_between(range(len(obs)), 
                    np.array(obs)[:, i].flatten() - np.array(meas_var)[:, i].flatten(), 
                    np.array(obs)[:, i].flatten() + np.array(meas_var)[:, i].flatten(), 
                    color='lightblue', alpha=0.2, label='Measurement Uncertainty')
    plt.title(f'Kalman Filter Estimate vs Observations for {var}')
    plt.xlabel('Time Step')
    plt.ylabel(var)
    plt.legend()
    plt.show()
