import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp

file_path=r'gdp\products_RS41-GDP-1_POT_2025\POT-RS-01_2_RS41-GDP_001_20250109T162800_1-000-001.nc'
gdp=gp.read(file_path)
data=gp.parcel_method(gdp.data)
data=gp.potential_temperature_gradient(gdp.data)
data=gp.RH_gradient(gdp.data)
data=gp.richardson_number_method(gdp.data)
print(data)

plt.figure()

plt.subplot(1, 4, 1)
plt.plot(gdp.data['temp'], gdp.data['alt'])
plt.xlabel('Temperature (K)')
plt.ylabel('Altitude (m)')
plt.title('Temperature vs Altitude')
plt.grid(True)

plt.subplot(1, 4, 2)
plt.plot(gdp.data['virtual_potential_temp'], gdp.data['alt'])
plt.xlabel('Virtual Potential Temperature (K)')
plt.ylabel('Altitude (m)')
plt.title('Virtual Potential Temperature vs Altitude')
plt.grid(True)

plt.subplot(1, 4, 3)
plt.plot(gdp.data['rh'], gdp.data['alt'])
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Altitude (m)')
plt.title('RH vs Altitude')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(gdp.data['wspeed'], gdp.data['alt'])
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Altitude (m)')
plt.title('Wind Speed vs Altitude')
plt.grid(True)

plt.tight_layout()
plt.show()