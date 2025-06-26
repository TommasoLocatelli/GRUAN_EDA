import numpy as np

class Formulas:
    """
    A class that contains various atmospheric formulas.
    """

    def __init__(self):
        pass

    def virtual_temperature_from_temp_rh_press(self, temp, rh, press):
        """
        Calculate the virtual potential temperature from temperature, relative humidity, and pressure.
        See Wallace and Hobbs (2006) for the formula 3.16
        
        Parameters:
        temp (float or array-like): Temperature in Kelvin.
        rh (float or array-like): Relative humidity in percentage.
        press (float or array-like): Pressure in hPa.

        Returns:
        float or array-like: Virtual potential temperature in Kelvin.
        """
        # Convert relative humidity to a fraction
        rh_fraction = rh / 100.0
        # Calculate saturation vapor pressure
        es = 6.112 * np.exp((17.67 * (temp - 273.15)) / (temp - 29.65))
        # Calculate actual vapor pressure
        e = es * rh_fraction
        # Calculate virtual potential temperature
        theta_v = temp * (1 + 0.61 * e / press)
        return theta_v
    
    def potential_temperature(self, temp, press):
        """
        Calculate the potential temperature from temperature and pressure with Poisson's equation.
        
        Parameters:
        temp (float or array-like): Temperature in Kelvin.
        press (float or array-like): Pressure in hPa.

        Returns:
        float or array-like: Potential temperature in Kelvin.
        """
        theta = temp * (1000 / press) ** 0.2869