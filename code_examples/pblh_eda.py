import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp


folder = r'gdp\products_RS41-GDP-1_POT_2025'
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:5]:
    gdp = gp.read(file_path)
    gdp.data = gdp.data[gdp.data['alt'] <= 10000]  # Limit to first 10 km for speed
    #gdp.data = gdp.data[gdp.data.index % 5 == 0]  # Thin data for speed
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
    # Look at PBLH methods variable of interest profiles: virtual potential temp, gradients, Ri
    data = gp.parcel_method(gdp.data)
    data = gp.potential_temperature_gradient(gdp.data, virtual=True)
    data = gp.RH_gradient(gdp.data)
    data = gp.bulk_richardson_number_method(gdp.data)
    pblh_pm = gdp.data['alt'][data['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data and any(data['pblh_pm'] == 1) else None
    pblh_theta = gdp.data['alt'][data['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data and any(data['pblh_theta'] == 1) else None
    pblh_rh = gdp.data['alt'][data['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data and any(data['pblh_rh'] == 1) else None
    pblh_Ri = gdp.data['alt'][data['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data and any(data['pblh_Ri'] == 1) else None


    # Look at thermodynamic and dynamic variable profiles
    plt.figure()
    plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)

    plt.subplot(2, 2, 1)
    # Plot temperature vs altitude with uncertainty rectangles
    for t, a, t_uc, a_uc in zip(gdp.data['temp'], gdp.data['alt'], gdp.data['temp_uc'], gdp.data['alt_uc']):
        rect = Ellipse((t, a), t_uc, a_uc, color='lightcoral', alpha=0.3)
        plt.gca().add_patch(rect)
    plt.scatter(gdp.data['temp'], gdp.data['alt'],
                #gdp.data['temp'][gdp.data.index % 2 == 0], gdp.data['alt'][gdp.data.index % 2 == 0], 
                color="#781E1A")
    #plt.axhline(pblh_theta, color='r', linestyle='--', label='PBLH_Theta')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (m)')
    plt.title('Temperature vs Altitude')# with Theta PBLH')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    # Plot RH vs altitude with uncertainty rectangles
    for rh, a, rh_uc, a_uc in zip(gdp.data['rh'], gdp.data['alt'], gdp.data['rh_uc'], gdp.data['alt_uc']):
        rect = Ellipse((rh, a), rh_uc, a_uc, color='lightskyblue', alpha=0.3)
        plt.gca().add_patch(rect)
    plt.scatter(gdp.data['rh'], gdp.data['alt'], color="#2C3993")
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Altitude (m)')
    plt.title('RH vs Altitude')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    # Plot wind speed vs altitude with uncertainty rectangles
    for ws, a, ws_uc, a_uc in zip(gdp.data['wspeed'], gdp.data['alt'], gdp.data['wspeed_uc'], gdp.data['alt_uc']):
        rect = Ellipse((ws, a), ws_uc, a_uc, color='lightgreen', alpha=0.3)
        plt.gca().add_patch(rect)
    plt.scatter(gdp.data['wspeed'], gdp.data['alt'], color="#329E32")
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Altitude (m)')
    plt.title('Wind Speed vs Altitude')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    # Plot pressure vs altitude with uncertainty rectangles
    for p, a, p_uc, a_uc in zip(gdp.data['press'], gdp.data['alt'], gdp.data['press_uc'], gdp.data['alt_uc']):
        rect = Ellipse((p, a), p_uc, a_uc, color='plum', alpha=0.3)
        plt.gca().add_patch(rect)
    plt.scatter(gdp.data['press'], gdp.data['alt'], color="#543395")
    plt.xlabel('Pressure (hPa)')
    plt.ylabel('Altitude (m)')
    plt.title('Pressure vs Altitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure()
    for es, a, es_uc, a_uc in zip(data['es'], gdp.data['alt'], gdp.data['es_uc'], gdp.data['alt_uc']):
        rect = Ellipse((es, a), es_uc, a_uc, color='lightgray', alpha=0.3)
        plt.gca().add_patch(rect)
    plt.scatter(data['es'], gdp.data['alt'], color="#781E1A")
    plt.xlabel('Saturation Vapor Pressure (hPa)')
    plt.ylabel('Altitude (m)')
    plt.title('Saturation Vapor Pressure vs Altitude')
    plt.grid(True)
    plt.show()
    break

    plt.figure()
    plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)
    plt.subplot(2, 2, 1)
    plt.plot(gdp.data['virtual_potential_temp'], gdp.data['alt'], color="#781E1A")
    plt.xlabel('Virtual Potential Temperature (K)')
    plt.ylabel('Altitude (m)')
    plt.title('Virtual Potential Temperature vs Altitude')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(data['potential_temp_gradient'], gdp.data['alt'], color="#2C3993")
    plt.xlabel('Theta Gradient (K/m)')
    plt.ylabel('Altitude (m)')
    plt.title('Potential Temperature Gradient vs Altitude')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(data['rh_gradient'], gdp.data['alt'], color="#329E32")
    plt.xlabel('RH Gradient (%/m)')
    plt.ylabel('Altitude (m)')
    plt.title('RH Gradient vs Altitude')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(gdp.data['Ri_b'], gdp.data['alt'], color="#543395")
    plt.xlabel('Richardson Number')
    plt.ylabel('Altitude (m)')
    plt.title('Richardson Number vs Altitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Look at PBLH results

    # how does the result change if gradient is modeled?

    # monte carlo simulation of PBLH methods proof of concept

    break  # Remove this break to process all files