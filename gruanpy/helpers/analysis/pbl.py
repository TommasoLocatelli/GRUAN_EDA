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
        # Consider only the first 5000 meters above the first altitude
        alt_limit = data['alt'].iloc[0] + 5000
        subset = data[data['alt'] <= alt_limit]
        index = subset[subset['virtual_temperature'] < surface_virtual_potential_temperature].index
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
        data['delta_temp'] = data['temp'].diff().fillna(0)
        data['delta_alt'] = data['alt'].diff().fillna(1)
        data['delta_alt'] = data['delta_alt'].apply(lambda x: x if abs(x) >= 1 else 1) #get rid of zero or very small values
        data['potential_temperature_gradient'] = data['delta_temp'] / data['delta_alt']
        data['pblh'] = 0
        # Find the index of the maximum gradient
        # Consider only the first 5000 meters above the first altitude
        alt_limit = data['alt'].iloc[0] + 5000
        subset = data[data['alt'] <= alt_limit]
        max_gradient_index = subset['potential_temperature_gradient'].idxmax()
        data.at[max_gradient_index, 'pblh'] = 1 
        return data
    
    def specific_humidity_gradient(self, data):
        """
        Calculate the specific humidity gradient and determine the PBLH
        as the altitude where the gradient is minimum.

        Parameters:
        data (DataFrame): Data containing 'specific_humidity' and 'alt' columns.

        Returns:
        DataFrame: Data with additional columns for specific humidity gradient
        and a marker for PBLH.
        """
        data['specific_humidity'] = ff.specific_humidity(
            data['temp'], data['rh'], data['press']
        )
        data['delta_sh'] = data['specific_humidity'].diff().fillna(0)
        data['delta_alt'] = data['alt'].diff().fillna(1)
        data['delta_alt'] = data['delta_alt'].apply(lambda x: x if abs(x) >= 1 else 1)
        data['specific_humidity_gradient'] = data['delta_sh'] / data['delta_alt']
        data['pblh'] = 0
        # Find the index of the minimum gradient    
        # Consider only the first 5000 meters above the first altitude
        alt_limit = data['alt'].iloc[0] + 5000
        subset = data[data['alt'] <= alt_limit]
        min_gradient_index = subset['specific_humidity_gradient'].idxmin()
        data.at[min_gradient_index, 'pblh'] = 1
        return data