# NetCDF File Information

## Basic Information
- **Filename**: api_response.nc
- **Path**: c:\Users\tomma\Documents\SDC\Repos\GRUAN_EDA\code_examples\gdp_demo_090625\api_response.nc
- **Size**: 4.49 MB

## Dimensions
- **valid_time**: 24
- **latitude**: 361
- **longitude**: 361

## Global Attributes
- **GRIB_centre**: ecmf
- **GRIB_centreDescription**: European Centre for Medium-Range Weather Forecasts
- **GRIB_subCentre**: 0
- **Conventions**: CF-1.7
- **institution**: European Centre for Medium-Range Weather Forecasts
- **history**: 2025-06-19T15:15 GRIB to CDM+CF via cfgrib-0.9.15.0/ecCodes-2.41.0 with {"source": "tmpaek7gegi/data.grib", "filter_by_keys": {"stream": ["oper"], "stepType": ["instant"]}, "encode_cf": ["parameter", "time", "geography", "vertical"]}

## Variables
### number
- **Type**: int64
- **Shape**: ()
- **Dimensions**: 
- **Attributes**:
  - long_name: ensemble member numerical id
  - units: 1
  - standard_name: realization

### valid_time
- **Type**: int64
- **Shape**: (24,)
- **Dimensions**: valid_time(24)
- **Attributes**:
  - long_name: time
  - standard_name: time
  - units: seconds since 1970-01-01
  - calendar: proleptic_gregorian

### latitude
- **Type**: float64
- **Shape**: (361,)
- **Dimensions**: latitude(361)
- **Attributes**:
  - _FillValue: nan
  - units: degrees_north
  - standard_name: latitude
  - long_name: latitude
  - stored_direction: decreasing

### longitude
- **Type**: float64
- **Shape**: (361,)
- **Dimensions**: longitude(361)
- **Attributes**:
  - _FillValue: nan
  - units: degrees_east
  - standard_name: longitude
  - long_name: longitude

### expver
- **Type**: <class 'str'>
- **Shape**: (24,)
- **Dimensions**: valid_time(24)

### t2m
- **Type**: float32
- **Shape**: (24, 361, 361)
- **Dimensions**: valid_time(24), latitude(361), longitude(361)
- **Attributes**:
  - _FillValue: nan
  - GRIB_paramId: 167
  - GRIB_dataType: an
  - GRIB_numberOfPoints: 130321
  - GRIB_typeOfLevel: surface
  - GRIB_stepUnits: 1
  - GRIB_stepType: instant
  - GRIB_gridType: regular_ll
  - GRIB_uvRelativeToGrid: 0
  - GRIB_NV: 0
  - GRIB_Nx: 361
  - GRIB_Ny: 361
  - GRIB_cfName: unknown
  - GRIB_cfVarName: t2m
  - GRIB_gridDefinitionDescription: Latitude/Longitude Grid
  - GRIB_iDirectionIncrementInDegrees: 0.25
  - GRIB_iScansNegatively: 0
  - GRIB_jDirectionIncrementInDegrees: 0.25
  - GRIB_jPointsAreConsecutive: 0
  - GRIB_jScansPositively: 0
  - GRIB_latitudeOfFirstGridPointInDegrees: 90.0
  - GRIB_latitudeOfLastGridPointInDegrees: 0.0
  - GRIB_longitudeOfFirstGridPointInDegrees: 0.0
  - GRIB_longitudeOfLastGridPointInDegrees: 90.0
  - GRIB_missingValue: 3.4028234663852886e+38
  - GRIB_name: 2 metre temperature
  - GRIB_shortName: 2t
  - GRIB_totalNumber: 0
  - GRIB_units: K
  - long_name: 2 metre temperature
  - units: K
  - standard_name: unknown
  - GRIB_surface: 0.0
  - coordinates: number valid_time latitude longitude expver

