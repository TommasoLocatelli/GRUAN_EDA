import gruanpy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------
def mwr_pre_proc(netcdf):
    data = netcdf.data

    # Filter out observations above 5000 m
    data = data[data["height"] <= 5000].copy()

    # Ensure time is datetime
    data['time'] = pd.to_datetime(data['time'])
    
    return data


# ---------------------------------------------------------
# Parcel method (time × height pivot)
# ---------------------------------------------------------
def mwr_parcel_method(theta_grid, offset=0.5):
    # theta_grid: index=time, columns=height
    times = theta_grid.index.values
    heights = theta_grid.columns.values

    parcel_pblh = []

    for t in times:
        profile = theta_grid.loc[t].values
        surf = profile[0]  # lowest height
        mask = profile > surf + offset

        if mask.any():
            idx = np.argmax(mask)
            height = heights[idx]
        else:
            height = np.nan

        parcel_pblh.append(height)

    return pd.DataFrame({
        "time": times,
        "pbl_height_parcel": parcel_pblh
    })


# ---------------------------------------------------------
# Gradient-based PBLH (time × height pivot)
# ---------------------------------------------------------
def mwr_pblh(data, pm_offset=0.5):

    # Potential temperature grid (time × height)
    theta_grid = data.pivot(index='time', columns='height', values='potential_temperature')

    # Compute vertical gradient
    heights = theta_grid.columns.values
    theta_grad = np.gradient(theta_grid.values, heights, axis=1)

    theta_grad_df = pd.DataFrame(theta_grad, index=theta_grid.index, columns=theta_grid.columns)

    # Thermal PBLH = max gradient
    pblh_theta_df = (
        theta_grad_df.apply(lambda row: heights[np.argmax(row.values)], axis=1)
        .rename("pbl_height_theta")
        .reset_index()
    )

    # Parcel method
    pblh_parcel_df = mwr_parcel_method(theta_grid, offset=pm_offset)

    # Relative humidity grid (time × height)
    rh_grid = data.pivot(index='time', columns='height', values='relative_humidity')

    rh_grad = np.gradient(rh_grid.values, heights, axis=1)
    rh_grad_df = pd.DataFrame(rh_grad, index=rh_grid.index, columns=rh_grid.columns)

    # Moist PBLH = min RH gradient
    pblh_rh_df = (
        rh_grad_df.apply(lambda row: heights[np.argmin(row.values)], axis=1)
        .rename("pbl_height_rh")
        .reset_index()
    )

    return pblh_theta_df, pblh_parcel_df, pblh_rh_df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == '__main__':

    mwr_paths = [
        r'data\mwr_sunny_week\20260817_cabauw_hatpro-multi_46797fd6.nc',
        r'data\mwr_sunny_week\20260818_cabauw_hatpro-multi_46797fd6.nc',
        r'data\cloudnet-examples\20260816_cabauw_hatpro-multi_46797fd6.nc',
        r'data\cloudnet-examples\20260817_cabauw_hatpro-multi_46797fd6.nc',
        r'data\cloudnet-examples\20260817_cabauw_hatpro-multi_46797fd6.nc'
    ]

    path = mwr_paths[2]
    netcdf = gp.read_netcdf(path)

    data = mwr_pre_proc(netcdf)

    # Compute PBLH diagnostics
    pblh_theta_df, pblh_parcel_df, pblh_rh_df = mwr_pblh(data, pm_offset=0.5)

    # ---------------------------------------------------------
    # TWO SUBPLOTS: θ + RH
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # ---------------------------------
    # SUBPLOT 1 — θ + Thermal + Parcel PBLH
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

    ax.plot(
        pblh_parcel_df['time'],
        pblh_parcel_df['pbl_height_parcel'],
        color='blue',
        linewidth=2,
        label='Parcel PBL (θ crossing)'
    )

    fig.colorbar(sc2, ax=ax).set_label('Potential Temperature (K)')
    ax.set_ylabel('Altitude (m)')
    ax.set_xlabel('Time')
    ax.set_title('Potential Temperature with Thermal & Parcel PBL height')
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

    fig.colorbar(sc1, ax=ax).set_label('Relative Humidity (%)')
    ax.set_xlabel('Time')
    ax.set_title('RH profile with Moist PBL height')
    ax.legend()

    axes[0].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
