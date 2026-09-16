import gruanpy as gp
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

path=r'data\cloudnet-lidar-ceilometer\20260816_cabauw_chm15k_c07f533c.nc'
netcdf=gp.read_netcdf(path)
data=netcdf.data

# Pivot to time × height grid
vdop = data.pivot(index="time", columns="height", values="beta_raw")
# Filter heights up to 5000 m
vdop_5km = vdop.loc[:, vdop.columns <= 1600]

vdop_5km = vdop.loc[:, vdop.columns <= 1600]
print(vdop_5km.shape)

times = pd.to_datetime(vdop_5km.index)
heights = vdop_5km.columns.values

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

pcm = ax.pcolormesh(
    times,
    heights,
    vdop_5km.T,
    shading="auto",
    cmap="RdBu_r",
    vmin=0,
    vmax=1e-6
)

ax.set_ylabel("Height [m]")
ax.set_xlabel("Time")
ax.set_title("Ceilometer β_raw (0–5000 m)")

fig.colorbar(pcm, ax=ax, label="sr⁻¹ m⁻¹")
plt.tight_layout()
plt.show()
