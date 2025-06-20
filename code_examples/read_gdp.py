import sys
import os
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
import matplotlib.pyplot as plt

file_path=r'gdp\data_examples\LIN-RS-01_2_IMS-100-GDP_002_20220107T093400_1-002-001.nc'
gdp=gp.read(file_path)
print(gdp.global_attrs)
print(gdp.data)
print(gdp.variables_attrs)
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=gdp.data['lon'], 
    y=gdp.data['lat'], 
    z=gdp.data['alt'],
    mode='lines',
    name='Trajectory'
))

fig.update_layout(
    title='Trajectory Over Time',
    scene = dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Altitude'
    )
)

fig.show()

# temperature vs altitude
plt.figure(figsize=(10, 6))

sc = plt.scatter(gdp.data['temp'], gdp.data['alt'], c=gdp.data['press'], cmap='coolwarm', marker='o')
plt.colorbar(sc, label='Pressure '+gdp.variables_attrs[gdp.variables_attrs['variable']=='press']['units'].values[0])

plt.xlabel('Temperature '+gdp.variables_attrs[gdp.variables_attrs['variable']=='temp']['units'].values[0])
plt.ylabel('Altitude m')
plt.title('Altitude vs. Temperature with Pressure as Color')
plt.grid(True)
plt.show()