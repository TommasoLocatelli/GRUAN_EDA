import gruanpy as gp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
                upper_bound=4000,
                cols=["time", "height", "altitude", "v", "beta", "beta_raw"]):

    # Time window (with +30 min offset)
    start = data["time"].dt.normalize().iloc[0] + pd.Timedelta(hours=start_hour, minutes=30)
    end   = data["time"].dt.normalize().iloc[0] + pd.Timedelta(hours=end_hour, minutes=30)

    # Apply time filter
    data = data[(data["time"] >= start) & (data["time"] <= end)]

    # Apply height filter
    data = data[data["height"] < upper_bound]

    # Select columns
    data = data[cols]

    return data

def plot_data(data):
    # --- Doppler velocity time–height plot ---

    # Pivot to time × height grid
    vdop = data.pivot(index="time", columns="height", values="v")

    # Convert time to datetime (if not already)
    vdop.index = pd.to_datetime(vdop.index)

    times = vdop.index.values
    heights = vdop.columns.values

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Doppler velocity plot
    pcm = ax.pcolormesh(times, heights, vdop.T, shading="auto", cmap="RdBu_r")

    # Labels and title
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Time")
    ax.set_title("Doppler Velocity (v)")

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("v [m/s]")

    plt.tight_layout()
    plt.show()

def compute_pblh(data):
    """
    Compute PBLH as the height of the local minimum in vertical wind speed (v)
    for each time step.
    """
    pblh = []

    # Pivot to time × height grid
    vdop = data.pivot(index="time", columns="height", values="v")
    vdop.index = pd.to_datetime(vdop.index)

    for t, row in vdop.iterrows():
        v_profile = row.values
        heights = row.index.values

        # Remove NaNs
        mask = ~np.isnan(v_profile)
        v_clean = v_profile[mask]
        h_clean = heights[mask]

        if len(v_clean) == 0:
            pblh.append(np.nan)
            continue

        # Find local minima
        minima_idx = argrelextrema(v_clean, np.less)[0]

        if len(minima_idx) > 0:
            # Choose the strongest minimum (lowest v)
            min_idx = minima_idx[np.argmin(v_clean[minima_idx])]
        else:
            # Fallback: global minimum
            min_idx = np.argmin(v_clean)

        pblh.append(h_clean[min_idx])

    return pd.Series(pblh, index=vdop.index, name="PBLH")

def plot_pblh(data, pblh):
    """
    Plot Doppler velocity time–height field with PBLH overlay.
    """
    # Pivot
    vdop = data.pivot(index="time", columns="height", values="v")
    vdop.index = pd.to_datetime(vdop.index)

    times = vdop.index.values
    heights = vdop.columns.values

    fig, ax = plt.subplots(figsize=(14, 6))

    # Doppler velocity field
    pcm = ax.pcolormesh(times, heights, vdop.T, shading="auto", cmap="RdBu_r")
    cbar = fig.colorbar(pcm, ax=ax, label="v [m/s]")

    # Overlay PBLH
    ax.plot(pblh.index, pblh.values,
        #olor="black",
        linewidth=2.5, label="PBLH")
    #ax.scatter(pblh.index, pblh.values, color="yellow", edgecolor="black", s=40)

    # Formatting
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Time")
    ax.set_title("Doppler Velocity with PBLH")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    path=r'data\cloudnet-doppler-lidar\20260812_cabauw_wls200s_dca88604.nc'
    netcdf=gp.read_netcdf(path)
    data=netcdf.data

    data=filter_data(data,
                    start_hour=1,
                    end_hour=23)

    #describe_data(data)

    #plot_data(data)

    pblh = compute_pblh(data)
    plot_pblh(data, pblh)
