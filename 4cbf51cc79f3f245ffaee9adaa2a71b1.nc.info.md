# NetCDF File Information

## Basic Information
- **Filename**: 4cbf51cc79f3f245ffaee9adaa2a71b1.nc
- **Path**: c:\Users\tomma\Documents\SDC\Repos\GRUAN_EDA\4cbf51cc79f3f245ffaee9adaa2a71b1.nc
- **Size**: 0.02 MB

## Dimensions
- **valid_time**: 1
- **latitude**: 3
- **longitude**: 3

## Global Attributes
- **GRIB_centre**: ecmf
- **GRIB_centreDescription**: European Centre for Medium-Range Weather Forecasts
- **GRIB_subCentre**: 0
- **Conventions**: CF-1.7
- **institution**: European Centre for Medium-Range Weather Forecasts
- **history**: 2026-04-10T10:13 GRIB to CDM+CF via cfgrib-0.9.15.1/ecCodes-2.42.0 with {"source": "tmpjax_c4ik/data.grib", "filter_by_keys": {"stream": ["enda"], "stepType": ["instant"]}, "encode_cf": ["parameter", "time", "geography", "vertical"]}

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
- **Shape**: (1,)
- **Dimensions**: valid_time(1)
- **Attributes**:
  - long_name: time
  - standard_name: time
  - units: seconds since 1970-01-01
  - calendar: proleptic_gregorian

### latitude
- **Type**: float64
- **Shape**: (3,)
- **Dimensions**: latitude(3)
- **Attributes**:
  - _FillValue: nan
  - units: degrees_north
  - standard_name: latitude
  - long_name: latitude
  - stored_direction: decreasing

### longitude
- **Type**: float64
- **Shape**: (3,)
- **Dimensions**: longitude(3)
- **Attributes**:
  - _FillValue: nan
  - units: degrees_east
  - standard_name: longitude
  - long_name: longitude

### expver
- **Type**: <class 'str'>
- **Shape**: ()
- **Dimensions**: 

### blh
- **Type**: float32
- **Shape**: (1, 3, 3)
- **Dimensions**: valid_time(1), latitude(3), longitude(3)
- **Attributes**:
  - _FillValue: nan
  - GRIB_paramId: 159
  - GRIB_dataType: es
  - GRIB_numberOfPoints: 9
  - GRIB_typeOfLevel: surface
  - GRIB_stepUnits: 1
  - GRIB_stepType: instant
  - GRIB_gridType: regular_ll
  - GRIB_uvRelativeToGrid: 0
  - GRIB_NV: 0
  - GRIB_Nx: 3
  - GRIB_Ny: 3
  - GRIB_cfName: unknown
  - GRIB_cfVarName: blh
  - GRIB_gridDefinitionDescription: Latitude/Longitude Grid
  - GRIB_iDirectionIncrementInDegrees: 0.5
  - GRIB_iScansNegatively: 0
  - GRIB_jDirectionIncrementInDegrees: 0.5
  - GRIB_jPointsAreConsecutive: 0
  - GRIB_jScansPositively: 0
  - GRIB_latitudeOfFirstGridPointInDegrees: 53.0
  - GRIB_latitudeOfLastGridPointInDegrees: 52.0
  - GRIB_longitudeOfFirstGridPointInDegrees: 14.0
  - GRIB_longitudeOfLastGridPointInDegrees: 15.0
  - GRIB_missingValue: 3.4028234663852886e+38
  - GRIB_name: Boundary layer height
  - GRIB_shortName: blh
  - GRIB_totalNumber: 10
  - GRIB_units: m
  - long_name: Boundary layer height
  - units: m
  - standard_name: unknown
  - GRIB_surface: 0.0
  - coordinates: number valid_time latitude longitude expver

