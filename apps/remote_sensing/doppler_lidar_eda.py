import gruanpy as gp
import pandas as pd

path=r'data\cloudnet-doppler-lidar\20260812_cabauw_wls200s_dca88604.nc'
netcdf=gp.read_netcdf(path)
data=netcdf.data

print(data.shape)

import matplotlib.pyplot as plt
import pandas as pd

# Pivot to time × height grid
vdop = data.pivot(index="time", columns="height", values="v")

# Convert time to datetime
times = pd.to_datetime(vdop.index)
heights = vdop.columns.values

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Doppler velocity plot
pcm = ax.pcolormesh(times, heights, vdop.T, shading="auto", cmap="RdBu_r")
ax.set_ylabel("Height [m]")
ax.set_xlabel("Time")
ax.set_title("Doppler Velocity (v)")

# Colorbar
fig.colorbar(pcm, ax=ax, label="v [m/s]")

plt.tight_layout()
plt.show()
