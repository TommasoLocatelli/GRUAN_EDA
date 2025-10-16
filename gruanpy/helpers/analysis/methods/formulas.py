from .constants import Constants
import numpy as np

CNST=Constants()

class Formulas:
    """A class containing various atmospheric formulas for analysis.
    This class provides static methods to compute different atmospheric properties
    such as potential temperature, virtual temperature, saturation vapor pressure,
    mixing ratios, densities, and more.
    """
    def from_celsius_to_kelvin(self, T_c):
        """Convert temperature from Celsius to Kelvin.
        Parameters:
        T_c (float or array-like): Temperature in Celsius.
        Returns:
        float or array-like: Temperature in Kelvin.
        """
        T_k = T_c + 273.15
        return T_k

    def from_kelvin_to_celsius(self, T_k):
        """Convert temperature from Kelvin to Celsius.
        Parameters:
        T_k (float or array-like): Temperature in Kelvin.
        Returns:
        float or array-like: Temperature in Celsius.
        """
        T_c = T_k - 273.15
        return T_c

    def virtual_temperature(self, T, e, p):
        """Calculate the virtual temperature.
        Parameters:
        T (float or array-like): Temperature in Kelvin.
        e (float or array-like): Vapor pressure in hPa.
        p (float or array-like): Pressure in hPa.
        Returns:
        float or array-like: Virtual temperature in Kelvin.
        """
        Tv = T / (1 - (e / p)*(1 - CNST.EPSILON))
        return Tv
    
    def virtual_temperature_uncertainty(self, T, e, p, T_uc, e_uc, p_uc):
        """Calculate the uncertainty in virtual temperature.
        Parameters:
        T (float or array-like): Temperature in Kelvin.
        e (float or array-like): Vapor pressure in hPa.
        p (float or array-like): Pressure in hPa.
        T_uc (float or array-like): Uncertainty in temperature in Kelvin.
        e_uc (float or array-like): Uncertainty in vapor pressure in hPa.
        p_uc (float or array-like): Uncertainty in pressure in hPa.
        Returns:
        float or array-like: Uncertainty in virtual temperature in Kelvin.
        """
        dTv_dT = 1 / (1 - (e / p)*(1 - CNST.EPSILON))
        dTv_dp = T * e * (1 - CNST.EPSILON) / ( p - e * (1 - CNST.EPSILON))**2
        dTv_de = p * T * (1 - CNST.EPSILON) /  (p - e *(1 - CNST.EPSILON))**2
        Tv_uc = np.sqrt((dTv_dT * T_uc)**2 + (dTv_de * e_uc)**2 + (dTv_dp * p_uc)**2)
        return Tv_uc

    def potential_temperature(self, T, p, p0=1000.0):
        """Calculate the potential temperature thanks to the Poisson equation.
        Parameters:
        T (float or array-like): Temperature in Kelvin.
        p (float or array-like): Pressure in hPa.
        p0 (float): Reference pressure in hPa. Default is 1000 hPa.
        Returns:
        float or array-like: Potential temperature in Kelvin.
        """
        theta = T * (p0 / p) ** (CNST.Poisson_exponent)
        return theta
    
    def potential_temperature_uncertainty(self, T, p, T_uc, p_uc, p0=1000.0):
        """Calculate the uncertainty in potential temperature.
        Parameters:
        T (float or array-like): Temperature in Kelvin.
        p (float or array-like): Pressure in hPa.
        T_uc (float or array-like): Uncertainty in temperature in Kelvin.
        p_uc (float or array-like): Uncertainty in pressure in hPa.
        p0 (float): Reference pressure in hPa. Default is 1000 hPa.
        Returns:
        float or array-like: Uncertainty in potential temperature in Kelvin.
        """
        dtheta_dT = (p0 / p) ** (CNST.Poisson_exponent)
        dtheta_dp = - ((CNST.Poisson_exponent) * T * ((p0 / p) ** (CNST.Poisson_exponent))) / p
        theta_uc = np.sqrt((dtheta_dT * T_uc)**2 + (dtheta_dp * p_uc)**2)
        return theta_uc

    def tetens_equation(self, T):
        """Calculate the saturation vapor pressure using Tetens equation.
        Parameters:
        T (float or array-like): Temperature in Kelvin.
        Returns:
        float or array-like: Saturation vapor pressure in hPa.
        """
        es = 6.1078 * np.exp((17.27 * (T - 273.16)) / (T - 35.86))
        return es
        
    def saturation_vapor_pressure_uncertainty(self, T, T_uc):
        """Calculate the uncertainty in saturation vapor pressure using Tetens equation.
        Parameters:
        T (float or array-like): Temperature in Kelvin.
        T_uc (float or array-like): Uncertainty in temperature in Kelvin.
        Returns:
        float or array-like: Uncertainty in saturation vapor pressure in hPa.
        """
        es_uc = 6.1078 * np.exp((17.27 * (T - 273.16)) / (T - 35.86)) * (4098.17/(T - 35.86)**2) * T_uc
        return es_uc

    def water_vapor_saturation_mass(self, es, p):
        """Calculate the water vapor saturation mass (mixing ratio at saturation).
        Parameters:
        es (float or array-like): Saturation vapor pressure in hPa.
        p (float or array-like): Pressure in hPa.
        Returns:
        float or array-like: Water vapor saturation mass in kg/kg.
        """
        ws = CNST.EPSILON * es / (p - es)
        return ws

    def mixing_ratio_from_RH(self, RH, ws):
        """Calculate the mixing ratio from relative humidity and saturation mixing ratio.
        Parameters:
        RH (float or array-like): Relative humidity in percentage (0-100).
        ws (float or array-like): Saturation mixing ratio in kg/kg.
        Returns:
        float or array-like: Mixing ratio in kg/kg.
        """
        w = (RH / 100.0) * ws
        return w

    def water_vapor_pressure_from_RH(self, RH, es):
        """Calculate the water vapor pressure from relative humidity and saturation vapor pressure.
        Parameters:
        RH (float or array-like): Relative humidity in percentage (0-100).
        es (float or array-like): Saturation vapor pressure in hPa.
        Returns:
        float or array-like: Water vapor pressure in hPa.
        """
        e = (RH / 100.0) * es
        return e
    
    def water_vapor_pressure_uncertainty(self, RH, es, RH_uc, es_uc):
        """Calculate the uncertainty in water vapor pressure from uncertainties in RH and es.
        Parameters:
        RH (float or array-like): Relative humidity in percentage (0-100).
        es (float or array-like): Saturation vapor pressure in hPa.
        RH_uc (float or array-like): Uncertainty in relative humidity in percentage.
        es_uc (float or array-like): Uncertainty in saturation vapor pressure in hPa.
        Returns:
        float or array-like: Uncertainty in water vapor pressure in hPa.
        """
        e_uc = np.sqrt((es * RH_uc / 100.0) ** 2 + (RH * es_uc / 100.0) ** 2)
        return e_uc

    def density_of_dry_air(self, pd, T):
        """Calculate the density of dry air using the ideal gas law.
        Parameters:
        pd (float or array-like): Dry air partial pressure in hPa.
        T (float or array-like): Temperature in Kelvin.
        Returns:
        float or array-like: Density of dry air in kg/m³.
        """
        p_pa = pd * 100  # Convert hPa to Pa
        rho = p_pa / (CNST.R_DRY_AIR * T)
        return rho

    def density_of_water_vapor(self, e, T):
        """Calculate the density of water vapor using the ideal gas law.
        Parameters:
        e (float or array-like): Water vapor pressure in hPa.
        T (float or array-like): Temperature in Kelvin.
        Returns:
        float or array-like: Density of water vapor in kg/m³.
        """
        e_pa = e * 100  # Convert hPa to Pa
        rho_v = e_pa / (CNST.R_WATER_VAPOR * T)
        return rho_v

    def specific_humidity_from_mixing_ratio(self, w):
        """Calculate the specific humidity from mixing ratio.
        Parameters:
        w (float or array-like): Mixing ratio in kg/kg.
        Returns:
        float or array-like: Specific humidity in kg/kg.
        """
        q = w / (1 + w)
        return q
    
    def bulk_richardson_number(self, virtual_pot_temp_s, virtual_pot_temp, z, u, v, g=CNST.G0):
        """
        version of Seidel et al. (2012) using bulk Richardson number, but do not interpolate to mid-points.
        z is the height above the surface, u and v are the wind horizontal speed components at height z,
        """
        numerator = (g/virtual_pot_temp_s) * (virtual_pot_temp - virtual_pot_temp_s) * z
        denominator = u**2 + v**2
        Ri_b = numerator / denominator
        return Ri_b