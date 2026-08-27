import gruanpy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

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

path = mwr_paths[1]
netcdf = gp.read_netcdf(path)
data = netcdf.data
# Filter out observations above 5000 m
data = data[data["height"] <= 5000]


pt = data.pivot(index="time", columns="height", values="potential_temperature")
pt.index = pd.to_datetime(pt.index)

pt_clean = pt.dropna(axis=1, how='any')

print(pt.shape)

from sklearn.decomposition import PCA

N_COMPONENTS=5

pca = PCA(n_components=N_COMPONENTS)
scores = pca.fit_transform(pt_clean)

from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

mod = DynamicFactor(scores, k_factors=2, factor_order=2)
res = mod.fit(maxiter=500)
print(res.summary())

scores_smooth = scores.copy()
scores_smooth[:, :N_COMPONENTS] = res.fittedvalues[:, :N_COMPONENTS]

reconstructed = scores_smooth @ pca.components_ + pca.mean_

# ----------------------------------------
# Prepare matrices
# ----------------------------------------

orig = pt_clean.values              # (time, heights)
recon = reconstructed               # (time, heights)
diff = orig - recon                 # (time, heights)

times = pt_clean.index
heights = pt_clean.columns

# ----------------------------------------
# Plot original, reconstructed, difference
# ----------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

vmin = orig.min()
vmax = orig.max()
im0 = axes[0].pcolormesh(times, heights, orig.T,
                         shading='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)

im1 = axes[1].pcolormesh(times, heights, recon.T,
                         shading='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)

im2 = axes[2].pcolormesh(times, heights, diff.T,
                         shading='auto', cmap='coolwarm')


axes[0].set_title("Original Potential Temperature")
axes[0].set_ylabel("Height [m]")
fig.colorbar(im0, ax=axes[0], label="K")

axes[1].set_title("Reconstructed (State-Space Smoothed)")
axes[1].set_ylabel("Height [m]")
fig.colorbar(im1, ax=axes[1], label="K")

axes[2].set_title("Difference (Original - Reconstructed)")
axes[2].set_ylabel("Height [m]")
axes[2].set_xlabel("Time")
fig.colorbar(im2, ax=axes[2], label="K")

plt.tight_layout()
plt.show()
# ----------------------------------------
# PCA diagnostics (dynamic)
# ----------------------------------------

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

# Bar plot of explained variance
labels = [f"PC{i+1}" for i in range(N_COMPONENTS)]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(labels, pca.explained_variance_ratio_)
ax.set_title("PCA Explained Variance Ratio")
ax.set_ylabel("Fraction of variance")
plt.show()

# Plot PCA vertical modes
fig, axes = plt.subplots(N_COMPONENTS, 1, figsize=(10, 4*N_COMPONENTS), sharex=True)

for i in range(N_COMPONENTS):
    axes[i].plot(heights, pca.components_[i, :])
    axes[i].set_title(f"PCA Mode {i+1}")
    axes[i].set_ylabel("Loading")

axes[-1].set_xlabel("Height [m]")
plt.tight_layout()
plt.show()
