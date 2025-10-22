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
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]

    # Calculate PBLH using different methods
    data = gp.parcel_method(gdp.data)
    data = gp.potential_temperature_gradient(gdp.data, virtual=True)
    data = gp.RH_gradient(gdp.data)
    data = gp.bulk_richardson_number_method(gdp.data)
    pblh_pm = gdp.data['alt'][data['pblh_pm'] == 1].iloc[0] if 'pblh_pm' in data and any(data['pblh_pm'] == 1) else None
    pblh_theta = gdp.data['alt'][data['pblh_theta'] == 1].iloc[0] if 'pblh_theta' in data and any(data['pblh_theta'] == 1) else None
    pblh_rh = gdp.data['alt'][data['pblh_rh'] == 1].iloc[0] if 'pblh_rh' in data and any(data['pblh_rh'] == 1) else None
    pblh_Ri = gdp.data['alt'][data['pblh_Ri'] == 1].iloc[0] if 'pblh_Ri' in data and any(data['pblh_Ri'] == 1) else None
    #gdp.data = gdp.data[gdp.data.index % 10 == 0]  # Thin data for speed, the bottle necks are the plots

    # Look at thermodynamic and dynamic variable profiles and uncertainties
    if False:
        plt.figure()
        plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)

        plt.subplot(2, 3, 1)
        # Plot temperature vs altitude with uncertainty ellipses
        for t, a, t_uc, a_uc in zip(gdp.data['temp'], gdp.data['alt'], gdp.data['temp_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((t, a), t_uc, a_uc, color='lightcoral', alpha=0.3)
            plt.gca().add_patch(ellip)
        for t, a, t_uc, a_uc in zip(gdp.data['virtual_potential_temp'], gdp.data['alt'], gdp.data['theta_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((t, a), t_uc, a_uc, color='gold', alpha=0.3)
            plt.gca().add_patch(ellip)
        plt.scatter(gdp.data['temp'], gdp.data['alt'],
                    #gdp.data['temp'][gdp.data.index % 2 == 0], gdp.data['alt'][gdp.data.index % 2 == 0], 
                    color="#781E1A", label='temp')
        #plt.axhline(pblh_theta, color='r', linestyle='--', label='PBLH_Theta')
        plt.scatter(gdp.data['virtual_potential_temp'], gdp.data['alt'], color='orange', label='theta')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Altitude (m)')
        plt.title('Temperature vs Altitude')# with Theta PBLH')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 3, 2)
        # Plot RH vs altitude with uncertainty ellipses
        for rh, a, rh_uc, a_uc in zip(gdp.data['rh'], gdp.data['alt'], gdp.data['rh_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((rh, a), rh_uc, a_uc, color='lightskyblue', alpha=0.3)
            plt.gca().add_patch(ellip)
        plt.scatter(gdp.data['rh'], gdp.data['alt'], color="#2C3993", label='rh')
        plt.xlabel('Relative Humidity (%)')
        plt.ylabel('Altitude (m)')
        plt.title('RH vs Altitude')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 3, 3)
        # Plot pressure vs altitude with uncertainty ellipses
        for p, a, p_uc, a_uc in zip(gdp.data['press'], gdp.data['alt'], gdp.data['press_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((p, a), p_uc, a_uc, color='plum', alpha=0.3)
            plt.gca().add_patch(ellip)
        for es, a, es_uc, a_uc in zip(gdp.data['es'], gdp.data['alt'], gdp.data['es_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((es, a), es_uc, a_uc, color='lightgray', alpha=0.3)
            plt.gca().add_patch(ellip)
        for e, a, e_uc, a_uc in zip(gdp.data['e'], gdp.data['alt'], gdp.data['e_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((e, a), e_uc, a_uc, color='violet', alpha=0.3)
            plt.gca().add_patch(ellip)
        plt.scatter(gdp.data['press'], gdp.data['alt'], color="#543395", label='press')
        plt.scatter(gdp.data['es'], gdp.data['alt'], color="#1A6878", label='es')
        plt.scatter(gdp.data['e'], gdp.data['alt'], color="#A41A8D", label='e')
        plt.xlabel('Pressure (hPa)')
        plt.ylabel('Altitude (m)')
        plt.title('Pressure vs Altitude')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 3, 4)
        # Plot potential temperature gradient vs altitude
        for ptg, a, ptg_uc, a_uc in zip(gdp.data['potential_temp_gradient'], gdp.data['alt'], gdp.data['potential_temp_gradient_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((ptg, a), ptg_uc, a_uc, color='salmon', alpha=0.3)
            plt.gca().add_patch(ellip)
        plt.scatter(gdp.data['potential_temp_gradient'], gdp.data['alt'], color="#932C3A", label='theta grad')
        plt.xlabel('Theta Gradient (K/m)')
        plt.ylabel('Altitude (m)')
        plt.title('Potential Temperature Gradient vs Altitude')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 3, 5)
        # Plot RH gradient vs altitude
        for rhg, a, rhg_uc, a_uc in zip(gdp.data['rh_gradient'], gdp.data['alt'], gdp.data['rh_gradient_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((rhg, a), 
            rhg_uc, a_uc, color='lightblue', alpha=0.3)
            plt.gca().add_patch(ellip)
        plt.scatter(gdp.data['rh_gradient'], gdp.data['alt'], color="#32859E", label='RH grad')
        plt.xlabel('RH Gradient (%/m)')
        plt.ylabel('Altitude (m)')
        plt.title('RH Gradient vs Altitude')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 3, 6)
        # Plot wind speed vs altitude with uncertainty ellipses
        for ws, a, ws_uc, a_uc in zip(gdp.data['wspeed'], gdp.data['alt'], gdp.data['wspeed_uc'], gdp.data['alt_uc']):
            ellip = Ellipse((ws, a), ws_uc, a_uc, color='lightgreen', alpha=0.3)
            plt.gca().add_patch(ellip)
        plt.scatter(gdp.data['wspeed'], gdp.data['alt'], color="#329E32", label='wspeed')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Altitude (m)')
        plt.title('Wind Speed vs Altitude')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Look at PBLH gradient results and uncertainties
    if True:
        plt.figure()
        plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)
        
        plt.subplot(1, 2, 1)
        # plot pot temp gradient with PBLH indication
        plt.errorbar(gdp.data['potential_temp_gradient'], gdp.data['alt'],
                        xerr=gdp.data['potential_temp_gradient_uc'], fmt='o', color="#932C3A", ecolor='lightcoral', alpha=0.7, label='theta grad')
        if pblh_theta is not None:
            plt.axhline(pblh_theta, color='r', linestyle='--', label='PBLH_Theta')
        plt.xlabel('Theta Gradient (K/m)')
        plt.ylabel('Altitude (m)')
        plt.title('Potential Temperature Gradient vs Altitude with PBLH')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        # plot RH gradient with PBLH indication
        plt.errorbar(gdp.data['rh_gradient'], gdp.data['alt'],
                        xerr=gdp.data['rh_gradient_uc'], fmt='o', color="#32859E", ecolor='lightskyblue', alpha=0.7, label='RH grad')
        if pblh_rh is not None:
            plt.axhline(pblh_rh, color='b', linestyle='--', label='PBLH_RH')
        plt.xlabel('RH Gradient (%/m)')
        plt.ylabel('Altitude (m)')
        plt.title('RH Gradient vs Altitude with PBLH')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    #Look at gradient uncertainties
    if True:
        plt.figure()
        plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)

        plt.subplot(1, 2, 1)
        # plot pot temp gradient uncertainty
        plt.scatter(gdp.data['potential_temp_gradient_uc'], gdp.data['potential_temp_gradient'], color="#D2691E", label='theta grad uc')
        plt.ylabel('Theta Gradient (K/m)')
        plt.xlabel('Theta Gradient Uncertainty (K/m)')
        plt.title('Potential Temperature Gradient Uncertainty')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # plot RH gradient uncertainty
        plt.scatter(gdp.data['rh_gradient_uc'], gdp.data['rh_gradient'], color="#20B2AA", label='RH grad uc')
        plt.ylabel('RH Gradient (%/m)')
        plt.xlabel('RH Gradient Uncertainty (%/m)')
        plt.title('RH Gradient Uncertainty')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)
        

    # Look at data['virtual_potential_temp'], data['alt'], data['uspeed'], data['vspeed'], data['Ri_b'] profiles and PBLH_Ri
    if False:
        plt.figure()
        plt.suptitle(f'GRUAN Profile: {where}, {when}', fontsize=16)

        plt.subplot(1, 3, 1)
        # plot virtual potential temperature and virtual potential temperature minus at the surface
        plt.scatter(gdp.data['virtual_potential_temp'], gdp.data['alt'], color="#781E1A", label='theta_v')
        theta_v_surface = gdp.data['virtual_potential_temp'].iloc[0]
        plt.scatter(gdp.data['virtual_potential_temp'] - theta_v_surface, gdp.data['alt'], color='orange', label='theta_v - theta_v_surface')
        plt.xlabel('Virtual Potential Temperature (K)')
        plt.ylabel('Altitude (m)')
        plt.title('Virtual Potential Temperature vs Altitude')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 3, 2)
        # plot uspeed, wspeed and u**2 + v**2
        plt.scatter(gdp.data['uspeed'], gdp.data['alt'], color="#2C3993", label='u speed')
        plt.scatter(gdp.data['vspeed'], gdp.data['alt'], color="#1A6878", label='v speed')
        plt.scatter(gdp.data['uspeed']**2 + gdp.data['vspeed']**2, gdp.data['alt'], color="#329E32", label='u^2 + v^2')
        plt.xlabel('Wind Speed Components (m/s)')
        plt.ylabel('Altitude (m)')
        plt.title('Wind Speed Components vs Altitude')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 3, 3)
        # plot Richardson number vs altitude with PBLH indication
        plt.scatter(gdp.data['Ri_b'], gdp.data['alt'], color="#711D9B", label='Ri_b')
        if pblh_Ri is not None:
            plt.axhline(pblh_Ri, color='r', linestyle='--', label='PBLH_Ri')
        plt.axvline(0.25, color='gray', linestyle=':', label='Ri=0.25')
        plt.xlabel('Bulk Richardson Number')
        plt.ylabel('Altitude (m)')
        plt.title('Bulk Richardson Number vs Altitude with PBLH')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    # how does the result change if gradient is modeled?

    # monte carlo simulation of PBLH methods proof of concept

    break  # Remove this break to process all files