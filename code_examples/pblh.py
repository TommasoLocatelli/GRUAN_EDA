import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp

file_path=r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250109T162800_1-000-001.nc'
file_path = r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250116T164200_1-000-001.nc'
#file_patg = r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250123T112900_1-000-001.nc'
#file_path = r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250130T164200_1-000-001.nc'
gdp=gp.read(file_path)

if False:
    data=gp.parcel_method(gdp.data)
    column_of_interest = 'virtual_temperature'

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(data[column_of_interest], data['alt'], marker='o')
    plt.xlabel(f'{column_of_interest}')
    plt.ylabel('Altitude m')
    if 'pblh' in data.columns:
        pblh_altitudes = data.loc[data['pblh'] == 1, 'alt']
        for alt in pblh_altitudes:
            plt.axhline(y=alt, color='r', linestyle='--', label='PBLH')
        if not pblh_altitudes.empty:
            plt.legend()
    plt.title(f'Altitude vs. {column_of_interest} ')
    plt.grid(True)
    plt.show()

if False:
    data=gp.potential_temperature_gradient(gdp.data)
    column_of_interest = 'potential_temperature_gradient'

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(data[column_of_interest], data['alt'], marker='o')
    plt.xlabel(f'{column_of_interest}')
    plt.ylabel('Altitude m')
    if 'pblh' in data.columns:
        pblh_altitudes = data.loc[data['pblh'] == 1, 'alt']
        for alt in pblh_altitudes:
            plt.axhline(y=alt, color='r', linestyle='--', label='PBLH')
        if not pblh_altitudes.empty:
            plt.legend()
    plt.title(f'Altitude vs. {column_of_interest} ')
    plt.grid(True)
    plt.show()

if False:
    data=gp.specific_humidity_gradient(gdp.data)
    column_of_interest = 'specific_humidity_gradient'

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(data[column_of_interest], data['alt'], marker='o')
    plt.xlabel(f'{column_of_interest}')
    plt.ylabel('Altitude m')
    if 'pblh' in data.columns:
        pblh_altitudes = data.loc[data['pblh'] == 1, 'alt']
        for alt in pblh_altitudes:
            plt.axhline(y=alt, color='r', linestyle='--', label='PBLH')
        if not pblh_altitudes.empty:
            plt.legend()
    plt.title(f'Altitude vs. {column_of_interest} ')
    plt.grid(True)
    plt.show()

if True:
    data=gp.parcel_method(gdp.data)
    parcel_pblh= data.loc[data['pblh'] == 1, 'alt'].values
    data=gp.potential_temperature_gradient(data)
    pot_temp_pblh= data.loc[data['pblh'] == 1, 'alt'].values
    data=gp.specific_humidity_gradient(data)
    specific_humidity_pblh= data.loc[data['pblh'] == 1, 'alt'].values

    plt.figure(figsize=(10, 6))
    plt.plot(data['temp'], data['alt'], label='Temperature (K)')
    plt.plot(data['rh'], data['alt'], label='Relative Humidity (%)')
    plt.xlabel('Value')
    plt.ylabel('Altitude (m)')
    for alt in parcel_pblh:
        plt.axhline(y=alt, color='r', linestyle='--', label='Parcel Method PBLH')
    for alt in pot_temp_pblh:
        plt.axhline(y=alt, color='g', linestyle='-.', label='Potential Temperature Gradient PBLH')
    for alt in specific_humidity_pblh:
        plt.axhline(y=alt, color='b', linestyle=':', label='Specific Humidity Gradient PBLH')
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title('Temperature and Relative Humidity Profiles')
    plt.legend()
    plt.grid(True)
    plt.show()
