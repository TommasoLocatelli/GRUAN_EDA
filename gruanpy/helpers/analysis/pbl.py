from .formulas import Formulas 
ff= Formulas()

class PBLHMethods:
    """
    A class that provides methods to calculate the planetary boundary layer height (PBLH)
    using different methods such as the parcel method and potential temperature gradient.
    """

    def __init__(self):
        pass
        
    def parcel_method(self, data): # To be checked
        """
        Calculate the planetary boundary layer height (PBLH) using the parcel method.
        This method identifies the height at which the virtual potential temperature
        drops below the surface virtual potential temperature.
        Parameters:
        data (DataFrame): Data containing 'temp', 'rh', and 'press' columns.
        Returns:
        DataFrame: Data with an additional column for PBLH.
        """
        # Calculate virtual potential temperature from temperature, relative humidity, and pressure
        data['virtual_temperature'] = ff.virtual_temperature_from_temp_rh_press(
            data['temp'], data['rh'], data['press']
            )
        # Identify the PBLH based on the parcel method
        # The PBLH is the height where the virtual potential temperature drops below the surface value
        data['pblh'] = 0
        surface_virtual_potential_temperature = data['virtual_temperature'].iloc[0]
        index = data[data['virtual_temperature'] < surface_virtual_potential_temperature].index
        if not index.empty:
            pblh_index = index[0]
            data.at[pblh_index, 'pblh'] = 1
        return data

    def potential_temperature_gradient(self, data):
        """
        Calculate the potential temperature gradient from the data and determine the PBLH
        as the altitude where the gradient is maximum.

        Parameters:
        data (DataFrame): Data containing 'temp', 'press', and 'alt' columns.

        Returns:
        DataFrame: Data with additional columns for potential temperature, its gradient,
        and a marker for PBLH.
        """
        # Compute potential temperature and its gradient
        data['potential_temperature'] = ff.potential_temperature(data['temp'], data['press'])
        # Drop rows with missing altitude or potential temperature values to avoid TypeError
        data = data.dropna(subset=['alt', 'potential_temperature']).reset_index(drop=True)
        data['potential_temperature_gradient'] = data['potential_temperature'].diff() / data['alt'].diff()
        data['pblh'] = 0
        # Find the index of the maximum gradient
        max_gradient_index = data['potential_temperature_gradient'].idxmax()
        data.at[max_gradient_index, 'pblh'] = 1 
        return data