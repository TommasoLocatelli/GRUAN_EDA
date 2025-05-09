import sys
import os
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp

file_path=r'gdp\NYA-RS-01_2_RS-11G-GDP_001_20200319T000000_1-002-001.nc'
gdp=gp.read(file_path)
print(gdp.global_attrs)
print(gdp.data.head())
print(gdp.variables_attrs.head())

#trajectory
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