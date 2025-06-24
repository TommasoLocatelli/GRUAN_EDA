def potential_temperature(temp, press):
    """
    Calculate the virtual potential temperature using Poisson's equation.
    See Wallace and Hobbs, Poisson's equation 3.54.
    
    Parameters:
    temp (float or array-like): Temperature in Kelvin.
    press (float or array-like): Pressure in hPa.
    
    Returns:
    float or array-like: Virtual potential temperature.
    """
    return temp * ((1000 / press) ** 0.286)

def virtual_potential_temperature(temp, mixing_ratio_water_vapor, mixing_ratio_dry_air):
    """
    Calculate the virtual potential temperature using the mixing ratios of water vapor and dry air.
    See https://glossary.ametsoc.org/wiki/Virtual_potential_temperature.
    
    Parameters:
    temp (float or array-like): Temperature in Kelvin.
    mixing_ratio_water_vapor (float or array-like): Mixing ratio of water vapor in kg/kg.
    mixing_ratio_dry_air (float or array-like): Mixing ratio of dry air in kg/kg.

    Returns:
    float or array-like: Virtual potential temperature.
    """
    return temp * (1 + 0.61 * mixing_ratio_water_vapor - mixing_ratio_dry_air)

def mixing_ratio_water_vapor(press, vapor_pressure):
    """
    Calculate the mixing ratio of water vapor.
    See https://glossary.ametsoc.org/wiki/Mixing_ratio

    Parameters:
    press (float or array-like): Pressure in hPa.
    vapor_pressure (float or array-like): Vapor pressure in hPa.
    
    Returns:
    float or array-like: Mixing ratio of water vapor in kg/kg.
    """
    return (0.622 * vapor_pressure) / (press - vapor_pressure)

