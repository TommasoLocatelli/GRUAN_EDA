# Copernicus Climate Data Store (CDS) API

# How to set up:
# https://cds.climate.copernicus.eu/how-to-api

# Ho to generate request:
# https://cds.climate.copernicus.eu/datasets/insitu-observations-gruan-reference-network?tab=overview

from gruanpy import gruanpy as gp
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

api_request = """
import cdsapi

dataset = "insitu-observations-gruan-reference-network"
request = {
    "variable": [
        "air_temperature",
        "air_pressure",
        "altitude",
        "geopotential_height"
    ],
    "year": "2018",
    "month": "02",
    "day": ["02"],
    "data_format": "netcdf"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
"""

#gp.exec_request(api_request)

file_path=r'821d8bfc3c29386363bf47e24d40bec1.nc'
gdp=gp.read(file_path)
print(gdp.global_attrs)
print(gdp.data.head())
print(gdp.variables_attrs.head())
print(gdp.data.columns)
"""
file_path=r'821d8bfc3c29386363bf47e24d40bec1.nc'
gdp=gp.read(file_path)

# Extract longitude and latitude data
longitude = gdp.data['longitude|station_configuration']
latitude = gdp.data['latitude|station_configuration']
# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(longitude, latitude)]
geo_df = gpd.GeoDataFrame(gdp.data, geometry=geometry)
# Set the coordinate reference system (CRS) to WGS84
gdf = gpd.read_file(
    "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
)
# Plot the data
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the world map (gdf)
gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
# Plot the points (geo_df)
geo_df.plot(ax=ax, color='red', markersize=50, label='Stations')

# Add legend and title
plt.legend()
plt.title("Station Locations on World Map")
plt.show()

"""