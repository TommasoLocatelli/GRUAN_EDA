import gruanpy as gp
import pandas as pd

path=r'data\cloudnet-examples\20260816_cabauw_wls200s-wind_dca88604.nc'
netcdf=gp.read_netcdf(path)
data=netcdf.data

print(data.shape)

import matplotlib.pyplot as plt
import numpy as np

# Prepare pivot tables: time × height grids
uw = data.pivot(index="time", columns="height", values="uwind")
vw = data.pivot(index="time", columns="height", values="vwind")

# Convert time to matplotlib datetime
times = pd.to_datetime(uw.index)
heights = uw.columns.values

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- U wind subplot ---
pcm1 = axes[0].pcolormesh(times, heights, uw.T, shading="auto", cmap="coolwarm")
axes[0].set_ylabel("Height [m]")
axes[0].set_title("U-wind")
fig.colorbar(pcm1, ax=axes[0], label="U-wind [m/s]")

# --- V wind subplot ---
pcm2 = axes[1].pcolormesh(times, heights, vw.T, shading="auto", cmap="coolwarm")
axes[1].set_ylabel("Height [m]")
axes[1].set_title("V-wind")
fig.colorbar(pcm2, ax=axes[1], label="V-wind [m/s]")

axes[1].set_xlabel("Time")

plt.tight_layout()
plt.show()
