import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import plotly.graph_objects as go


# Load data
df = pd.read_csv(r"data/solar_baloon/solarballoon_09july2026.csv")
#df = pd.read_csv(r"data/solar_baloon/solarballoon_19june2026.csv")

# Convert timestamp column to datetime
# Try 'datetime' first, fall back to 'time_received'
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    time_col = "datetime"
else:
    df["time_received"] = pd.to_datetime(df["time_received"], errors="coerce")
    time_col = "time_received"

# Drop rows with invalid timestamps
df = df.dropna(subset=[time_col])

# Compute flight duration
start_time = df[time_col].min()
end_time   = df[time_col].max()
duration   = end_time - start_time

print("Flight start:", start_time)
print("Flight end:  ", end_time)
print("Flight duration:", duration)

# Extract coordinates
lats = df["lat"]
lons = df["lon"]
alts = df["alt"]

# 3D plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

ax.plot(lons, lats, alts, color="blue", linewidth=1)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude (m)")
ax.set_title("Solar Balloon 3D Trajectory – 09 July 2026")

plt.show()

# World map
fig = plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Set global extent (zoomed out)
ax.set_global()

# Plot trajectory
ax.plot(lons, lats, '-', color='red', linewidth=1, transform=ccrs.PlateCarree())

plt.title("Solar Balloon Trajectory – 09 July 2026")
plt.show()
