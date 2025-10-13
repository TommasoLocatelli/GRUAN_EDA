import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp


file_paths = [
    r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250109T162800_1-000-001.nc'
    ,r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250116T164200_1-000-001.nc'
    ,r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250123T112900_1-000-001.nc'
    ,r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250123T160600_1-000-001.nc'
    ,r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250130T113000_1-000-001.nc'
]

for file_path in file_paths:
    gdp=gp.read(file_path)
    data=gp.parcel_method(gdp.data)
    data=gp.potential_temperature_gradient(gdp.data)
    data=gp.RH_gradient(gdp.data)
    data=gp.bulk_richardson_number_method(gdp.data)

    plt.figure()
    where=gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when=gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
    plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)

    plt.subplot(2, 3, 1)
    plt.plot(gdp.data['temp'], gdp.data['alt'])
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (m)')
    plt.title('Temperature vs Altitude')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(gdp.data['virtual_temp'], gdp.data['alt'])
    plt.xlabel('Virtual Temperature (K)')
    plt.ylabel('Altitude (m)')
    plt.title('Virtual Temperature vs Altitude')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(gdp.data['virtual_potential_temp'], gdp.data['alt'])
    plt.xlabel('Virtual Potential Temperature (K)')
    plt.ylabel('Altitude (m)')
    plt.title('Virtual Potential Temperature vs Altitude')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(gdp.data['rh'], gdp.data['alt'])
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Altitude (m)')
    plt.title('RH vs Altitude')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(gdp.data['wspeed'], gdp.data['alt'])
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Altitude (m)')
    plt.title('Wind Speed vs Altitude')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(gdp.data['Ri_b'], gdp.data['alt'])
    plt.xlabel('Richardson Number')
    plt.ylabel('Altitude (m)')
    plt.title('Richardson Number vs Altitude')
    plt.grid(True)

    # Add horizontal lines for PBLH from different methods
    pblh_pm = gdp.data['alt'][data['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data and any(data['pblh_pm'] == 1) else None
    pblh_theta = gdp.data['alt'][data['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data and any(data['pblh_theta'] == 1) else None
    pblh_rh = gdp.data['alt'][data['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data and any(data['pblh_rh'] == 1) else None
    pblh_Ri = gdp.data['alt'][data['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data and any(data['pblh_Ri'] == 1) else None

    pblh_values = [pblh_pm, pblh_theta, pblh_rh, pblh_Ri]
    pblh_colors = ['r', 'g', 'b', 'm']
    pblh_labels = ['PBLH_PM', 'PBLH_Theta', 'PBLH_RH', 'PBLH_Ri']

    for i in range(1, 7):
        plt.subplot(2, 3, i)
        for pblh, color, label in zip(pblh_values, pblh_colors, pblh_labels):
            if pblh is not None:
                plt.axhline(
                    y=pblh,
                    color=color,
                    linestyle=['--', '-.', ':', (0, (5, 10))][pblh_labels.index(label) % 4],
                    linewidth=2,
                    label=label,
                    alpha=0.7
                )
        if i == 1:
            plt.legend(loc='best')

    plt.tight_layout()
    plt.show()