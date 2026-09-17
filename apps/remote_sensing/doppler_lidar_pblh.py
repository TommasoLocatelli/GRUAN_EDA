import gruanpy as gp
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

def describe_data(data):
    print("Shape:", data.shape)

    print("\nColumns:", data.columns.tolist())

    print("\nColumn types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isna().sum())

    print("\nStats:")
    print(data.describe(include='all'))


def filter_data(data, 
                start_hour=12, 
                end_hour=13,
                cols= ["time", "height", "altitude", "v", "beta", "beta_raw"]):
    start = data["time"].dt.normalize().iloc[0] + pd.Timedelta(hours=start_hour)
    end   = data["time"].dt.normalize().iloc[0] + pd.Timedelta(hours=end_hour)
    data = data[(data["time"] >= start) & (data["time"] <= end)]
    data = data[cols]
    return data
    

path=r'data\cloudnet-doppler-lidar\20260812_cabauw_wls200s_dca88604.nc'
netcdf=gp.read_netcdf(path)
data=netcdf.data
#describe_data(data)
#data["time"] = pd.to_datetime(data["time"], errors="coerce")

data=filter_data(data)

describe_data(data)