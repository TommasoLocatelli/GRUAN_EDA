# Copernicus Climate Data Store (CDS) API

# How to set up:
# https://cds.climate.copernicus.eu/how-to-api

# Ho to generate request:
# https://cds.climate.copernicus.eu/datasets/insitu-observations-gruan-reference-network?tab=overview

import cdsapi

dataset = "insitu-observations-gruan-reference-network"
request = {
    "variable": [
        "air_temperature",
        "relative_humidity",
        "relative_humidity_effective_vertical_resolution",
        "wind_speed",
        "wind_from_direction",
        "eastward_wind_speed",
        "northward_wind_speed",
        "shortwave_radiation",
        "air_pressure",
        "altitude",
        "geopotential_height",
        "frost_point_temperature",
        "water_vapour_volume_mixing_ratio",
        "vertical_speed_of_radiosonde",
        "time_since_launch"
    ],
    "year": "2017",
    "month": "04",
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15"
    ],
    "data_format": "netcdf"
}

#client = cdsapi.Client()
#client.retrieve(dataset, request).download()

from gruanpy import gruanpy as gp

file_path=r'a28f5ee8204f2b21f6100a702d0e99e5.nc'
gdp=gp.read(file_path)
print(gdp.global_attrs)
print(gdp.data.head())
print(gdp.variables_attrs.head())