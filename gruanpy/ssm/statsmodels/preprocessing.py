import numpy as np
from gruanpy.physics.formulas import virtual_potential_temperature, virtual_potential_temperature_uncertainty

def data_prep(data):
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

    Thv     = virtual_potential_temperature(T, p, r)
    Thv_unc  = virtual_potential_temperature_uncertainty(T, p, r, T_unc, p_unc, r_unc,)
    Thv_var = (Thv_unc * 0.5)**2

    # -----------------------------
    # FIT MODEL
    # ----------------------------- 
    endog=np.column_stack([z, Thv, RH, u, v])
    measurement_var = np.column_stack([z_var, Thv_var, RH_var, u_var, v_var]).T

    return endog, measurement_var
