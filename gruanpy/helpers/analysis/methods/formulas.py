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
        Tv = T * (1 - (e / p)*(1 - CNST.EPSILON))
        return Tv

    def potential_temperature(self, T, p, p0=1000.0):
        """Calculate the potential temperature thanks to the Poisson equation.
        Parameters:
        T (float or array-like): Temperature in Kelvin.
        p (float or array-like): Pressure in hPa.
        p0 (float): Reference pressure in hPa. Default is 1000 hPa.
        Returns:
        float or array-like: Potential temperature in Kelvin.
        """
        theta = T * (p0 / p) ** (CNST.R_DRY_AIR / CNST.C_P_DRY_AIR)
        return theta

    def tetens_equation(self, T):
        """Calculate the saturation vapor pressure using Tetens equation.
        Parameters:
        T (float or array-like): Temperature in Kelvin.
        Returns:
        float or array-like: Saturation vapor pressure in hPa.
        """
        T_C = self.from_kelvin_to_celsius(T)
        es = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
        return es

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

    def richardson_number(self, avg_Tv, delta_virtual_pot_temp, delta_z, delta_u, delta_v, g=CNST.G0):
        """Calculate the bulk Richardson number.
        Parameters:
        avg_Tv (float or array-like): Average virtual temperature in Kelvin.
        delta_virtual_pot_temp (float or array-like): Difference in virtual potential temperature in Kelvin.
        delta_z (float or array-like): Difference in height in meters.
        delta_u (float or array-like): Difference in u-component of wind in m/s.
        delta_v (float or array-like): Difference in v-component of wind in m/s.
        Returns:
        float or array-like: Bulk Richardson number (dimensionless).
        """
        if True:
            print('Deprecated: use bulk_richardson_number instead.')
        else:
            numerator = g * (delta_virtual_pot_temp / avg_Tv) * delta_z
            denominator = delta_u**2 + delta_v**2
            Ri_b = numerator / denominator
            return Ri_b
    
    def bulk_richardson_number(self, virtual_pot_temp_s, virtual_pot_temp, z, u, v, g=CNST.G0):
        """
        version of Seidel et al. (2012) using bulk Richardson number, but do not interpolate to mid-points.
        z is the height above the surface, u and v are the wind horizontal speed components at height z,
        """
        numerator = (g/virtual_pot_temp_s) * (virtual_pot_temp - virtual_pot_temp_s) * z
        denominator = u**2 + v**2
        Ri_b = numerator / denominator
        return Ri_b