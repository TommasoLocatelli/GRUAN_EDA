import numpy as np
from .formulas import Formulas
FM=Formulas()

class PBLHMethods:
    """
    A class that contains various methods to estimate the Planetary Boundary Layer Height (PBLH).
    Each method implements a different criterion for determining the PBLH based on atmospheric data.
    """

    def _find_upper_bound(self, data, upper_bound=5000):
        ground_level = data['alt'].min()
        self.altitude_bound = ground_level + upper_bound

    def parcel_method(self, data):
        """
        "Mixing height based on hypothetical vertical displacement of a parcel of air 
        from the surface and identification of the height at which virtual potential temperature 
        is equal to the surface value." Seidel et al. (2010)"
        """
        self._find_upper_bound(data)
        # computes missing variables if not present
        data['es']=FM.tetens_equation(data['temp']) if 'es' not in data else data['es']
        data['e']=FM.water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
        data['virtual_temp']=FM.virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
        data['virtual_potential_temp']=FM.potential_temperature(data['virtual_temp'], data['press']) if 'virtual_potential_temp' not in data else data['virtual_potential_temp']
        # apply parcel method criterion
        data['pblh_pm'] = 0
        surface_virtual_potential_temperature = data['virtual_potential_temp'].iloc[0]
        index = data[(data['virtual_potential_temp'] < surface_virtual_potential_temperature) & (data['alt'] <= self.altitude_bound)].index
        if not index.empty:
            pblh_index = index[0]
            data.at[pblh_index, 'pblh_pm'] = 1
        else:
            data.at[data.index[0], 'pblh_pm'] = 1  # If no crossing found, set to the lowest level
        return data

    def potential_temperature_gradient(self, data, virtual=False):
        """
        "Location of the maximum vertical gradient of potential temperature." 
        Seidel et al. (2010)
        The virtual flag indicates whether to use virtual potential temperature
        (if True) or potential temperature (if False).
        Vertical gradient is computed using finite differences.
        """
        self._find_upper_bound(data)
        # computes missing variables if not present
        if virtual:
            temp_clmn='virtual_potential_temp'
            data['es']=FM.tetens_equation(data['temp']) if 'es' not in data else data['es']
            data['e']=FM.water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
            data['virtual_temp']=FM.virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
            data[temp_clmn]=FM.potential_temperature(data['virtual_temp'], data['press']) if 'potential_temp' not in data else data['potential_temp']
        else:
            temp_clmn='potential_temp'
            data[temp_clmn]=FM.potential_temperature(data['temp'], data['press']) if 'potential_temp' not in data else data['potential_temp']
        # compute gradient and apply criterion
        data['potential_temp_gradient'] = (data[temp_clmn].diff() / data['alt'].diff())
        data['pblh_theta'] = 0
        max_gradient_index = data[(data['alt'] <= self.altitude_bound)]['potential_temp_gradient'].idxmax()
        data.at[max_gradient_index, 'pblh_theta'] = 1 
        return data

    def RH_gradient(self, data):
        """
        "Location of the minimum vertical gradient of relative humidity " 
        Seidel et al. (2010)
        Vertical gradient is computed using finite differences.
        """
        self._find_upper_bound(data)
        data['rh_gradient'] = (data['rh'].diff() / data['alt'].diff())
        data['pblh_rh'] = 0
        min_gradient_index = data[(data['alt'] <= self.altitude_bound)]['rh_gradient'].idxmin()
        data.at[min_gradient_index, 'pblh_rh'] = 1 
        return data

    def richardson_number_method(self, data):
        """
        "The ratio of buoyancy-related turbulence to mechanical shear-related turbulence 
        is calculated to obtain the Richardson number, which determines the PBLH as the lowest level 
        when Ri crosses a critical value of 0.25" Li et al. (2021)
        """
        self._find_upper_bound(data)
        # computes missing variables if not present
        data['es']=FM.tetens_equation(data['temp']) if 'es' not in data else data['es']
        data['e']=FM.water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
        data['virtual_temp']=FM.virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
        data['virtual_potential_temp']=FM.potential_temperature(data['virtual_temp'], data['press']) if 'virtual_potential_temp' not in data else data['virtual_potential_temp']
        data['uspeed'] = data['wspeed'] * np.cos(np.radians(data['wdir'])) if 'uspeed' not in data else data['uspeed']
        data['vspeed'] = data['wspeed'] * np.sin(np.radians(data['wdir'])) if 'vspeed' not in data else data['vspeed']
        # compute gradients
        data['delta_virtual_potential_temp'] = data['virtual_potential_temp'].diff()
        data['delta_z'] = data['alt'].diff()
        data['avg_Tv'] = (data['virtual_temp'] + data['virtual_temp'].shift(1)) / 2
        data['delta_u'] = data['uspeed'].diff()
        data['delta_v'] = data['vspeed'].diff()
        # compute Richardson number
        data['Ri_b'] = FM.bulk_richardson_number(data['avg_Tv'], data['delta_virtual_potential_temp'], data['delta_z'], data['delta_u'], data['delta_v'])
        # apply criterion
        data['pblh_Ri'] = 0
        index = data[(data['Ri_b'] > 0.25) & (data['alt'] <= self.altitude_bound)].index
        if not index.empty:
            pblh_index = index[0]
            data.at[pblh_index, 'pblh_Ri'] = 1
        return data

    def bulk_richardson_number_method(self, data):
        """
        version of Seidel et al. (2012) using bulk Richardson number
        """
        self._find_upper_bound(data)
        # computes missing variables if not present
        data['es']=FM.tetens_equation(data['temp']) if 'es' not in data else data['es']
        data['e']=FM.water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
        data['virtual_temp']=FM.virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
        data['virtual_potential_temp']=FM.potential_temperature(data['virtual_temp'], data['press']) if 'virtual_potential_temp' not in data else data['virtual_potential_temp']
        data['uspeed'] = data['wspeed'] * np.cos(np.radians(data['wdir'])) if 'uspeed' not in data else data['uspeed']
        data['vspeed'] = data['wspeed'] * np.sin(np.radians(data['wdir'])) if 'vspeed' not in data else data['vspeed']
        # compute Richardson number
        surface_virtual_potential_temperature = data['virtual_potential_temp'].iloc[0]
        data['Ri_b'] = FM.bulk_richardson_number(surface_virtual_potential_temperature, data['virtual_potential_temp'], data['alt'], data['uspeed'], data['vspeed'])
        # apply criterion
        data['pblh_Ri'] = 0
        index = data[(data['Ri_b'] > 0.25) & (data['alt'] <= self.altitude_bound)].index
        if not index.empty:
            pblh_index = index[0]
            data.at[pblh_index, 'pblh_Ri'] = 1
        return data
