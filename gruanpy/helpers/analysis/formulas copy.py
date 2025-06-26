import numpy as np

class Formulas:
    """
    A class that contains various atmospheric formulas.
    """

    def __init__(self):
        pass

    def virtual_potential_temperature(self, theta, r, rl):
        """
        Calculate the virtual potential temperature from potential temperature and mixing ratio.
        See https://en.wikipedia.org/wiki/Potential_temperature#Virtual_potential_temperature
        
        Parameters:
        theta (float or array-like): Potential temperature in Kelvin.
        r (float or array-like): Mixing ratio of water vapor in kg/kg.
        rl (float or array-like): Mixing ratio of dry air in kg/kg.

        Returns:
        float or array-like: Virtual potential temperature.
        """
        theta_v = theta * (1 + 0.61 * r - rl)
        return theta_v

    def potential_temperature(self, temp, press):
        """
        Calculate the potential temperature from temperature and pressure with Poisson's equation.
        See https://en.wikipedia.org/wiki/Potential_temperature

        Parameters:
        temp (float or array-like): Temperature in Kelvin.
        press (float or array-like): Pressure in hPa.

        Returns:
        float or array-like: Potential temperature in Kelvin.
        """
        theta = temp * (1000 / press) ** 0.2869  # Using Rd/cp = 0.286 for dry air
        return theta

    def saturation_vapor_pressure(self, temp): # To be checked
        """
        Calculate the saturation vapor pressure from temperature.
        See https://en.wikipedia.org/wiki/Vapor_pressure#Saturation_vapor_pressure

        Parameters:
        temp (float or array-like): Temperature in Kelvin.

        Returns:
        float or array-like: Saturation vapor pressure in hPa.
        """
        es = 6.112 * np.exp((17.67 * (temp - 273.15)) / (temp - 29.65))
        return es

    def actual_vapor_pressure_from_saturation(self, es): # To be checked
        """    Calculate the actual vapor pressure from saturation vapor pressure.
        See https://en.wikipedia.org/wiki/Vapor_pressure#Actual_vapor_pressure
        Parameters:
        es (float or array-like): Saturation vapor pressure in hPa.
        Returns:
        float or array-like: Actual vapor pressure in hPa.
        """
        e = es  # Assuming actual vapor pressure equals saturation vapor pressure
        return e

    def mixing_ratio_from_actual_vapor_pressure(self, press, e): # To be checked
        """
        Calculate the mixing ratio of water vapor from pressure and actual vapor pressure.
        See https://en.wikipedia.org/wiki/Mixing_ratio#Definition

        Parameters:
        press (float or array-like): Pressure in hPa.
        e (float or array-like): Actual vapor pressure in hPa.

        Returns:
        float or array-like: Mixing ratio of water vapor in kg/kg.
        """
        r = (0.622 * e) / (press - e)
        return r

    def liquid_water_mixing_ratio(self, press, r): # To be checked
        """
        Calculate the mixing ratio of dry air from pressure and mixing ratio of water vapor.
        See https://en.wikipedia.org/wiki/Mixing_ratio#Definition

        Parameters:
        press (float or array-like): Pressure in hPa.
        r (float or array-like): Mixing ratio of water vapor in kg/kg.

        Returns:
        float or array-like: Mixing ratio of dry air in kg/kg.
        """
        rl = press / (0.622 + r)
        return rl

    def virtual_potential_temperature_from_temp_rh_press(self, temp, rh, press):
        """
        Calculate the virtual potential temperature from temperature, relative humidity, and pressure.
        Parameters:
        temp (float or array-like): Temperature in Kelvin.
        rh (float or array-like): Relative humidity in percent (0-100).
        press (float or array-like): Pressure in hPa.

        Returns:
        float or array-like: Virtual potential temperature in Kelvin.
        """
        es = self.saturation_vapor_pressure(temp)
        e = self.actual_vapor_pressure_from_saturation(es)
        r = self.mixing_ratio_from_actual_vapor_pressure(press, e)
        rl = self.liquid_water_mixing_ratio(press, r)
        theta = self.potential_temperature(temp, press)
        theta_v = self.virtual_potential_temperature(theta, r, rl)
        return theta_v