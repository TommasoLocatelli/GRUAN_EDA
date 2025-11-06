import numpy as np
from .formulas import Formulas
FM=Formulas()

class PBLHMethods:
    """
    A class that contains various methods to estimate the Planetary Boundary Layer Height (PBLH).
    Each method implements a different criterion for determining the PBLH based on atmospheric data.
    """

    def _find_upper_bound(self, data, upper_bound=3500, return_value=False):
        ground_level = data['alt'].min()
        self.altitude_bound = ground_level + upper_bound
        if return_value:
            return self.altitude_bound

    def parcel_method(self, data):
        """
        "Mixing height based on hypothetical vertical displacement of a parcel of air 
        from the surface and identification of the height at which virtual potential temperature 
        is equal to the surface value." Seidel et al. (2010)"
        """
        self._find_upper_bound(data)
        # computes missing variables if not present
        data['es']=FM.tetens_equation(data['temp']) if 'es' not in data else data['es']
        data['es_uc']=FM.saturation_vapor_pressure_uncertainty(data['temp'], data['temp_uc']) if 'es' not in data else data['es']
        data['e']=FM.water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
        data['e_uc']=FM.water_vapor_pressure_uncertainty(data['rh'], data['es'], data['rh_uc'], data['es_uc']) if 'e_uc' not in data else data['e_uc']
        data['virtual_temp']=FM.virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
        data['virtual_temp_uc']=FM.virtual_temperature_uncertainty(data['temp'], data['e'], data['press'], data['temp_uc'], data['e_uc'], data['press_uc']) if 'virtual_temp_uc' not in data else data['virtual_temp_uc']
        data['virtual_theta']=FM.potential_temperature(data['virtual_temp'], data['press']) if 'virtual_theta' not in data else data['virtual_theta']
        data['virtual_theta_uc']=FM.potential_temperature_uncertainty(data['virtual_temp'], data['press'], data['virtual_temp_uc'], data['press_uc']) if 'theta_uc' not in data else data['theta_uc']
        # apply parcel method criterion
        data['pblh_pm'] = 0
        surface_virtual_potential_temperature = data['virtual_theta'].iloc[0]
        index = data[(data['virtual_theta'] < surface_virtual_potential_temperature) & (data['alt'] <= self.altitude_bound)].index
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
            temp_clmn='virtual_theta'
            data['es']=FM.tetens_equation(data['temp']) if 'es' not in data else data['es']
            data['es_uc']=FM.saturation_vapor_pressure_uncertainty(data['temp'], data['temp_uc']) if 'es' not in data else data['es']
            data['e']=FM.water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
            data['e_uc']=FM.water_vapor_pressure_uncertainty(data['rh'], data['es'], data['rh_uc'], data['es_uc']) if 'e_uc' not in data else data['e_uc']
            data['virtual_temp']=FM.virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
            data['virtual_temp_uc']=FM.virtual_temperature_uncertainty(data['temp'], data['e'], data['press'], data['temp_uc'], data['e_uc'], data['press_uc']) if 'virtual_temp_uc' not in data else data['virtual_temp_uc']
            data[temp_clmn]=FM.potential_temperature(data['virtual_temp'], data['press']) if 'theta' not in data else data['theta']
            data[temp_clmn+'_uc']=FM.potential_temperature_uncertainty(data['virtual_temp'], data['press'], data['virtual_temp_uc'], data['press_uc']) if temp_clmn+'_uc' not in data else data[temp_clmn+'_uc']
        else:
            temp_clmn='theta'
            data[temp_clmn]=FM.potential_temperature(data['temp'], data['press']) if 'theta' not in data else data['theta']
            data[temp_clmn+'_uc']=FM.potential_temperature_uncertainty(data['temp'], data['press'], data['temp_uc'], data['press_uc']) if temp_clmn+'_uc' not in data else data[temp_clmn+'_uc']
        # compute gradient and apply criterion
        data['theta_gradient'] = FM.finite_difference_gradient(data[temp_clmn], data['alt'])
        data['theta_gradient_uc'] = FM.finite_difference_gradient_uncertainty(data[temp_clmn], data['alt'], data[temp_clmn+'_uc'], data['alt_uc'])
        data['pblh_theta'] = 0
        max_gradient_index = data[(data['alt'] <= self.altitude_bound) & (data.index > 1)]['theta_gradient'].idxmax()
        data.at[max_gradient_index, 'pblh_theta'] = 1 
        return data

    def RH_gradient(self, data):
        """
        "Location of the minimum vertical gradient of relative humidity " 
        Seidel et al. (2010)
        Vertical gradient is computed using finite differences.
        """
        self._find_upper_bound(data)
        data['rh_gradient'] = FM.finite_difference_gradient(data['rh'], data['alt'])
        data['rh_gradient_uc'] = FM.finite_difference_gradient_uncertainty(data['rh'], data['alt'], data['rh_uc'], data['alt_uc'])
        # apply criterion
        data['pblh_rh'] = 0
        min_gradient_index = data[(data['alt'] <= self.altitude_bound)]['rh_gradient'].idxmin()
        data.at[min_gradient_index, 'pblh_rh'] = 1 
        return data

    def bulk_richardson_number_method(self, data):
        """
        version of Seidel et al. (2012) using bulk Richardson number
        """
        self._find_upper_bound(data)
        # computes missing variables if not present
        data['es']=FM.tetens_equation(data['temp']) if 'es' not in data else data['es']
        data['es_uc']=FM.saturation_vapor_pressure_uncertainty(data['temp'], data['temp_uc']) if 'es' not in data else data['es']
        data['e']=FM.water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
        data['virtual_temp']=FM.virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
        data['virtual_theta']=FM.potential_temperature(data['virtual_temp'], data['press']) if 'virtual_theta' not in data else data['virtual_theta']
        data['uspeed'] = data['wspeed'] * np.cos(np.radians(data['wdir'])) if 'uspeed' not in data else data['uspeed']
        data['vspeed'] = data['wspeed'] * np.sin(np.radians(data['wdir'])) if 'vspeed' not in data else data['vspeed']
        # compute Richardson number
        surface_virtual_potential_temperature = data['virtual_theta'].iloc[0]
        data['Ri_b'] = FM.bulk_richardson_number(surface_virtual_potential_temperature, data['virtual_theta'], data['alt'], data['uspeed'], data['vspeed'])
        # apply criterion
        data['pblh_Ri'] = 0
        index = data[(data['Ri_b'] > 0.25) & (data['alt'] <= self.altitude_bound)].index
        if not index.empty:
            pblh_index = index[0]
            data.at[pblh_index, 'pblh_Ri'] = 1
        return data
