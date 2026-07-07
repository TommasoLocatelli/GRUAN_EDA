import numpy as np
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
from ssm.statsmodels.pretrasformed_local_trend import PreTransformedLocalLinearTrend

def fit_ssm(gdp, method='lbfgs', iterations=100):
    data = gdp.data

    # -----------------------------
    # TIME
    # -----------------------------
    start = data['time'].values[0]
    time = data['time'].values
    seconds = (time - start) / np.timedelta64(1, 's')
    seconds = seconds.astype(float)

    # -----------------------------
    # ALTITUDE
    # -----------------------------
    z = data['alt'].values.astype(float)
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

    # -----------------------------
    # FIT MODEL
    # ----------------------------- 
    endog=np.column_stack([z, Thv, RH, u, v])
    measurement_var = np.column_stack([z_var, Thv_var, RH_var, u_var, v_var]).T

    # Setup the model
    model = PreTransformedLocalLinearTrend(endog=endog, measurement_var=measurement_var)

    # Fit it using MLE with a fixed sequence of measurement variances
    results = model.fit(method=method,
                maxiter=iterations,
                full_output=1,
                disp=5)
    
    return model, results

def simulate_ssm(model, M, seed=42):
    simulator=model.simulation_smoother(seed=seed) # default method is KFS; (method='cfa')  # can specify CFA method
    simulations=[]
    for _ in range(M):
        simulator.simulate()
        simulated_state=simulator.simulated_state
        simulations.append(simulated_state)
    return simulations

def smooth_variables(smoothed_state, smoothed_state_cov=None):

    # ----------------------------------------------------
    # Extract smoothed states
    # ----------------------------------------------------
    sm_alt     = smoothed_state[0]
    sm_alt_rc  = smoothed_state[1]

    sm_thv     = smoothed_state[2]
    sm_thv_rc  = smoothed_state[3]

    sm_rh      = smoothed_state[4]
    sm_rh_rc   = smoothed_state[5]

    sm_u       = smoothed_state[6]
    sm_u_rc    = smoothed_state[7]

    sm_v       = smoothed_state[8]
    sm_v_rc    = smoothed_state[9]

    # ----------------------------------------------------
    # Initialize uncertainties as None
    # (so simulation runs without covariance)
    # ----------------------------------------------------
    sm_alt_unc = sm_alt_rc_unc = None
    sm_thv_unc = sm_thv_rc_unc = None
    sm_rh_unc = sm_rh_rc_unc = None
    sm_u_unc = sm_u_rc_unc = None
    sm_v_unc = sm_v_rc_unc = None

    # ----------------------------------------------------
    # Extract smoothed-state uncertainties (2σ)
    # Only if covariance is provided
    # ----------------------------------------------------
    if smoothed_state_cov is not None:

        # Ensure positivity (Kalman numerical stability)
        cov = np.maximum(smoothed_state_cov, 0)

        sm_alt_unc    = np.sqrt(cov[0, 0, :]) * 2
        sm_alt_rc_unc = np.sqrt(cov[1, 1, :]) * 2

        sm_thv_unc    = np.sqrt(cov[2, 2, :]) * 2
        sm_thv_rc_unc = np.sqrt(cov[3, 3, :]) * 2

        sm_rh_unc     = np.sqrt(cov[4, 4, :]) * 2
        sm_rh_rc_unc  = np.sqrt(cov[5, 5, :]) * 2

        sm_u_unc      = np.sqrt(cov[6, 6, :]) * 2
        sm_u_rc_unc   = np.sqrt(cov[7, 7, :]) * 2

        sm_v_unc      = np.sqrt(cov[8, 8, :]) * 2
        sm_v_rc_unc   = np.sqrt(cov[9, 9, :]) * 2

    # ----------------------------------------------------
    # Compute smoothed diagnostic variables
    # ----------------------------------------------------
    sm_thv_grad = sm_thv_rc / sm_alt_rc
    sm_rh_grad  = sm_rh_rc  / sm_alt_rc

    # Gradient uncertainties only if covariances exist
    if smoothed_state_cov is not None:
        sm_thv_grad_uc = np.sqrt(
            (sm_thv_rc_unc / sm_alt_rc)**2 +
            (sm_thv_rc * sm_alt_rc_unc / sm_alt_rc**2)**2
        ) * 2

        sm_rh_grad_uc = np.sqrt(
            (sm_rh_rc_unc / sm_alt_rc)**2 +
            (sm_rh_rc * sm_alt_rc_unc / sm_alt_rc**2)**2
        ) * 2
    else:
        sm_thv_grad_uc = None
        sm_rh_grad_uc  = None

    # ----------------------------------------------------
    # Richardson number and uncertainty
    # ----------------------------------------------------
    g = 9.81

    sm_rich = ((g / sm_thv) * (sm_thv_rc / sm_alt_rc)) / (
        (sm_u_rc / sm_alt_rc)**2 + (sm_v_rc / sm_alt_rc)**2
    )

    if smoothed_state_cov is not None:

        thv_term = (g / sm_thv) * (sm_thv_rc / sm_alt_rc)
        wind_shear = (sm_u_rc / sm_alt_rc)**2 + (sm_v_rc / sm_alt_rc)**2

        dRi_dthv     = -thv_term / sm_thv / wind_shear
        dRi_dthv_rc  = (g / sm_thv) * (1 / sm_alt_rc) / wind_shear
        dRi_dalt_rc  = -(g / sm_thv) * (sm_thv_rc / sm_alt_rc**2) / wind_shear
        dRi_du_rc    = -thv_term * (2 * sm_u_rc / sm_alt_rc**2) / wind_shear**2
        dRi_dv_rc    = -thv_term * (2 * sm_v_rc / sm_alt_rc**2) / wind_shear**2

        sm_rich_uc = np.sqrt(
            (dRi_dthv    * sm_thv_unc)**2 +
            (dRi_dthv_rc * sm_thv_rc_unc)**2 +
            (dRi_dalt_rc * sm_alt_rc_unc)**2 +
            (dRi_du_rc   * sm_u_rc_unc)**2 +
            (dRi_dv_rc   * sm_v_rc_unc)**2
        )
    else:
        sm_rich_uc = None

    # ----------------------------------------------------
    # Return dictionary
    # ----------------------------------------------------
    return {
        "alt": sm_alt,
        "alt_rc": sm_alt_rc,
        "thv": sm_thv,
        "thv_rc": sm_thv_rc,
        "rh": sm_rh,
        "rh_rc": sm_rh_rc,
        "u": sm_u,
        "u_rc": sm_u_rc,
        "v": sm_v,
        "v_rc": sm_v_rc,

        "alt_unc": sm_alt_unc,
        "alt_rc_unc": sm_alt_rc_unc,
        "thv_unc": sm_thv_unc,
        "thv_rc_unc": sm_thv_rc_unc,
        "rh_unc": sm_rh_unc,
        "rh_rc_unc": sm_rh_rc_unc,
        "u_unc": sm_u_unc,
        "u_rc_unc": sm_u_rc_unc,
        "v_unc": sm_v_unc,
        "v_rc_unc": sm_v_rc_unc,

        "thv_grad": sm_thv_grad,
        "thv_grad_unc": sm_thv_grad_uc,
        "rh_grad": sm_rh_grad,
        "rh_grad_unc": sm_rh_grad_uc,

        "rich": sm_rich,
        "rich_unc": sm_rich_uc,
    }

def smooth_pblh(smooth_vars):

    # ----------------------------------------------------
    # Extract variables
    # ----------------------------------------------------
    alt       = smooth_vars["alt"]
    thv       = smooth_vars["thv"]
    thv_grad  = smooth_vars["thv_grad"]
    rh_grad   = smooth_vars["rh_grad"]
    rich      = smooth_vars["rich"]

    # ----------------------------------------------------
    # 1. Parcel Method (PM)
    # ----------------------------------------------------
    # PBLH = first height where θv(z) > θv(surface)
    thv_surf = thv[0]
    indices = np.where(thv > thv_surf)[0]

    if len(indices) > 0:
        pblh_pm = alt[indices[0]]
    else:
        pblh_pm = alt[0]

    # ----------------------------------------------------
    # 2. θv Gradient Minimum Method
    # ----------------------------------------------------
    idx_thv = np.argmax(thv_grad)
    pblh_thv = alt[idx_thv]

    # ----------------------------------------------------
    # 3. RH Gradient Minimum Method
    # ----------------------------------------------------
    idx_rh = np.argmin(rh_grad)
    pblh_rh = alt[idx_rh]

    # ----------------------------------------------------
    # 4. Richardson Number Method
    # ----------------------------------------------------
    # PBLH = first height where Ri > 0.25
    ri_threshold = 0.25
    indices = np.where(rich > ri_threshold)[0]

    if len(indices) > 0:
        pblh_ri = alt[indices[0]]
    else:
        pblh_ri = alt[0]


    # ----------------------------------------------------
    # Return dictionary
    # ----------------------------------------------------
    return {
        "pm":  pblh_pm,
        "thv": pblh_thv,
        "rh":  pblh_rh,
        "ri":  pblh_ri,
    }
