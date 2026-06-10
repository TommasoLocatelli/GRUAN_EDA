import numpy as np
import os
import sys
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp


def preprocess_profile(path, upper_bound=3000):
    """
    Preprocess a GRUAN GDP file and return:
        obs       : ndarray (n, 8)
        meas_var  : ndarray (n, 8)

    No normalization, no model components.
    Only raw observations and measurement variances.
    """

    # ---------------------------------------------------------
    # Load GRUAN GDP file
    # ---------------------------------------------------------
    gdp = gp.read(path)

    # determine PBLH upper bound
    upper_bound = gp._find_upper_bound(
        gdp.data[['alt']],
        upper_bound=upper_bound,
        return_value=True
    )

    # restrict to first part of profile
    data = gdp.data[gdp.data['alt'] <= upper_bound].copy()

    # ensure sorted by altitude
    data = data.sort_values("alt")

    # ---------------------------------------------------------
    # Extract observations
    # ---------------------------------------------------------
    z   = data['alt'].values
    Lz  = data['vspeed'].values
    T   = data['temp'].values
    p   = data['press'].values
    RH  = data['rh'].values
    r   = data['wvmr_mass'].values
    u   = data['wzon'].values
    v   = data['wmeri'].values

    # ---------------------------------------------------------
    # Extract variances (convert uncertainties to variances)
    # ---------------------------------------------------------
    z_var  = (data['alt_uc'].values * 0.5)**2
    Lz_var = (data['vspeed_uc'].values * 0.5)**2
    T_var  = (data['temp_uc'].values * 0.5)**2
    p_var  = (data['press_uc'].values * 0.5)**2
    RH_var = (data['rh_uc'].values * 0.5)**2
    r_var  = (data['wvmr_mass_uc'].values * 0.5)**2
    u_var  = (data['wzon_uc'].values * 0.5)**2
    v_var  = (data['wmeri_uc'].values * 0.5)**2

    # ---------------------------------------------------------
    # Build ndarray (n, 8)
    # ---------------------------------------------------------
    obs = np.column_stack([z, Lz, T, p, RH, r, u, v]).astype(np.float64)
    meas_var = np.column_stack([z_var, Lz_var, T_var, p_var, RH_var, r_var, u_var, v_var]).astype(np.float64)

    # ---------------------------------------------------------
    # NaN check
    # ---------------------------------------------------------
    if np.isnan(obs).any() or np.isnan(meas_var).any():
        warnings.warn("obs or meas_var contains NaN values")

    return obs, meas_var
