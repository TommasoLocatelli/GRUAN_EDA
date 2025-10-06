import numpy as np
from .formulas import *

def parcel_method(data):
    """
    "Mixing height based on hypothetical vertical displacement of a parcel of air 
    from the surface and identification of the height at which virtual potential temperature 
    is equal to the surface value." Seidel et al. (2010)"
    """
    # computes missing variables if not present
    data['es']=tetens_equation(data['temp']) if 'es' not in data else data['es']
    data['e']=water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
    data['virtual_temp']=virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
    data['virtual_potential_temp']=potential_temperature(data['virtual_temp'], data['press']) if 'virtual_potential_temp' not in data else data['virtual_potential_temp']
    # apply parcel method criterion
    data['pblh_pm'] = 0
    surface_virtual_potential_temperature = data['virtual_potential_temperature'].iloc[0]
    index = data[data['virtual_potential_temperature'] < surface_virtual_potential_temperature].index
    if not index.empty:
        pblh_index = index[0]
        data.at[pblh_index, 'pblh_pm'] = 1
    return data

def potential_temperature_gradient(data, virtual=False):
    """
    "Location of the maximum vertical gradient of potential temperature." 
    Seidel et al. (2010)
    The virtual flag indicates whether to use virtual potential temperature
    (if True) or potential temperature (if False).
    Vertical gradient is computed using finite differences.
    """
    # computes missing variables if not present
    if virtual:
        temp_clmn='virtual_potential_temp'
        data['es']=tetens_equation(data['temp']) if 'es' not in data else data['es']
        data['e']=water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
        data['virtual_temp']=virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
        data[temp_clmn]=potential_temperature(data['virtual_temp'], data['press']) if 'potential_temp' not in data else data['potential_temp']
    else:
        temp_clmn='potential_temp'
        data[temp_clmn]=potential_temperature(data['temp'], data['press']) if 'potential_temp' not in data else data['potential_temp']
    # compute gradient and apply criterion
    data['potential_temp_gradient'] = (data[temp_clmn].diff() / data['alt'].diff())
    data['pblh_theta'] = 0
    max_gradient_index = data['potential_temp_gradient'].idxmax()
    data.at[max_gradient_index, 'pblh_theta'] = 1 
    return data

def RH_gradient(data):
    """
    "Location of the minimum vertical gradient of relative humidity " 
    Seidel et al. (2010)
    Vertical gradient is computed using finite differences.
    """
    data['rh_gradient'] = (data['rh'].diff() / data['alt'].diff())
    data['pblh_rh'] = 0
    min_gradient_index = data['rh_gradient'].idxmin()
    data.at[min_gradient_index, 'pblh_rh'] = 1 
    return data

def richardson_number_method(data):
    """
    "The ratio of buoyancy-related turbulence to mechanical shear-related turbulence 
    is calculated to obtain the Richardson number, which determines the PBLH as the lowest level 
    when Ri crosses a critical value of 0.25" Li et al. (2021)
    """
    # computes missing variables if not present
    data['es']=tetens_equation(data['temp']) if 'es' not in data else data['es']
    data['e']=water_vapor_pressure_from_RH(data['rh'], data['es']) if 'e' not in data else data['e']
    data['virtual_temp']=virtual_temperature(data['temp'], data['e'], data['press']) if 'virtual_temp' not in data else data['virtual_temp']
    data['virtual_potential_temp']=potential_temperature(data['virtual_temp'], data['press']) if 'virtual_potential_temp' not in data else data['virtual_potential_temp']
    # compute gradients
    data['delta_virtual_potential_temp'] = data['virtual_potential_temp'].diff()
    data['delta_z'] = data['alt'].diff()
    data['delta_u'] = data['u'].diff() if 'u' in data else 0
    data['avg_Tv'] = (data['virtual_temp'] + data['virtual_temp'].shift(1)) / 2
    data['delta_wspeed'] = data['wspeed'].diff()
    # compute Richardson number
    data['Ri_b'] = bulk_richardson_number(data['avg_Tv'], data['delta_virtual_potential_temp'], data['delta_z'], data['delta_wspeed'])
    # apply criterion
    data['pblh_Ri'] = 0
    index = data[data['Ri'] > 0.25].index
    if not index.empty:
        pblh_index = index[0]
        data.at[pblh_index, 'pblh_Ri'] = 1
    return data
