import gruanpy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Cloudnet MWR product
# -----------------------------
mwr_paths = [
    r'data\mwr_sunny_week\20260817_cabauw_hatpro-multi_46797fd6.nc',
    r'data\mwr_sunny_week\20260818_cabauw_hatpro-multi_46797fd6.nc',
    r'data\cloudnet-examples\20260816_cabauw_hatpro-multi_46797fd6.nc',
    r'data\cloudnet-examples\20260817_cabauw_hatpro-multi_46797fd6.nc',
    r'data\cloudnet-examples\20260817_cabauw_hatpro-multi_46797fd6.nc'
]

path = mwr_paths[2]
netcdf = gp.read_netcdf(path)
data = netcdf.data

# Filter out observations above 5000 m
data = data[data["height"] <= 5000]

# Ensure time is datetime
data['time'] = pd.to_datetime(data['time'])

# -----------------------------
# RH GRADIENT & MOIST PBLH
# -----------------------------
grid_rh = data.pivot(index='height', columns='time', values='relative_humidity')

height_vals = grid_rh.index.values
rh_grad = np.gradient(grid_rh.values, height_vals, axis=0)

rh_grad_df = pd.DataFrame(rh_grad, index=grid_rh.index, columns=grid_rh.columns)

rh_grad_long = rh_grad_df.stack().reset_index()
rh_grad_long.columns = ['height', 'time', 'rh_gradient']

# Moist PBLH = minimum RH gradient
pblh_rh_df = (
    rh_grad_long.groupby('time')
    .apply(lambda g: g.loc[g['rh_gradient'].idxmin()][['height']], include_groups=False)
    .reset_index()
)
pblh_rh_df.columns = ['time', 'pbl_height_rh']

# -----------------------------
# POTENTIAL TEMPERATURE GRADIENT & THERMAL PBLH
# -----------------------------
theta_grid = data.pivot(index='height', columns='time', values='potential_temperature')

theta_grad = np.gradient(theta_grid.values, height_vals, axis=0)

theta_grad_df = pd.DataFrame(theta_grad, index=theta_grid.index, columns=theta_grid.columns)

theta_grad_long = theta_grad_df.stack().reset_index()
theta_grad_long.columns = ['height', 'time', 'theta_gradient']

# Thermal PBLH = maximum theta gradient
pblh_theta_df = (
    theta_grad_long.groupby('time')
    .apply(lambda g: g.loc[g['theta_gradient'].idxmax()][['height']], include_groups=False)
    .reset_index()
)
pblh_theta_df.columns = ['time', 'pbl_height_theta']

# -----------------------------
# PARCEL METHOD PBLH (θ crossing)
# -----------------------------
theta_surface = theta_grid.loc[theta_grid.index.min()]

parcel_pblh = []
for t in theta_grid.columns:
    profile = theta_grid[t].values
    surf = theta_surface[t]
    mask = profile > surf #+ 0.5
    if mask.any():
        idx = np.argmax(mask)
        height = theta_grid.index[idx]
    else:
        height = np.nan
    parcel_pblh.append(height)

pblh_parcel_df = pd.DataFrame({
    "time": theta_grid.columns,
    "pbl_height_parcel": parcel_pblh
})


# -----------------------------
# TWO SUBPLOTS: RH + θ
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# ---------------------------------
# SUBPLOT 1 — θ + Thermal PBLH
# ---------------------------------
ax = axes[0]

sc2 = ax.scatter(
    data['time'],
    data['height'],
    c=data['potential_temperature'],
    cmap='plasma',
    s=10
)

ax.plot(
    pblh_theta_df['time'],
    pblh_theta_df['pbl_height_theta'],
    color='black',
    linewidth=2,
    label='Thermal PBL (max θ gradient)'
)

cbar2 = fig.colorbar(sc2, ax=ax)
cbar2.set_label('Potential Temperature (K)')

# Parcel method PBLH
ax.plot(
    pblh_parcel_df['time'],
    pblh_parcel_df['pbl_height_parcel'],
    color='blue',
    linewidth=2,
    label='Parcel PBL (θ crossing)'
)


ax.set_xlabel('Time')
ax.set_title('Potential Temperature with Thermal PBL height')
ax.legend()

# ---------------------------------
# SUBPLOT 2 — RH + Moist PBLH
# ---------------------------------
ax = axes[1]

sc1 = ax.scatter(
    data['time'],
    data['height'],
    c=data['relative_humidity'],
    cmap='viridis_r',
    s=10
)

ax.plot(
    pblh_rh_df['time'],
    pblh_rh_df['pbl_height_rh'],
    color='red',
    linewidth=2,
    label='Moist PBL (min RH gradient)'
)

cbar1 = fig.colorbar(sc1, ax=ax)
cbar1.set_label('Relative Humidity (%)')

ax.set_xlabel('Time')
ax.set_ylabel('Altitude (m)')
ax.set_title('RH profile with Moist PBL height')
ax.legend()

axes[0].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
