import gruanpy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mwr_pblh import mwr_pre_proc, mwr_parcel_method
from mwr_pca_ssm import apply_pca, pca_diagnostic, reconstruct_observation, reconstruction_diagnostic, pblh_diagnostic
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL
from mwr_pblh_unc import simulate_ssm, reconstruct_simulations, pblh_monte_carlo_diagnostic


if __name__=='__main__':
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

    data = mwr_pre_proc(netcdf)

    pivot = data.pivot(index="time", columns="height", values="potential_temperature").dropna(axis=1, how='any')
    print(f"Observation dimensions {pivot.shape}")

    N_COMPONENTS=7

    means = pivot.values.mean(axis=0)
    stdevs = pivot.values.std(axis=0, ddof=0)
    X_std = (pivot.values - means) / stdevs
    pca, scores = apply_pca(X_std, N_COMPONENTS)
    print(f"PCA latent dimensions", scores.shape)

    #pca_diagnostic(pca, pivot, X_std)

    ssm=MultivariateLLL(scores)
    results=ssm.fit(maxiter=500)

    #print(results.summary())    

    M=50

    simulations=simulate_ssm(ssm, M, seed=42)
    print(f"Simulation dimension {simulations[0].shape}")
    
    reconstructed_sims = reconstruct_simulations(
        scores,
        simulations,
        pca=pca,
        means=means,
        stdevs=stdevs,
        n_components=N_COMPONENTS
    )
    print(f"Reconstruced simulations {reconstructed_sims[0].shape}")      # (T, n_heights)


    sim_pblh_list = []

    for recon in reconstructed_sims:
        recon_df = pd.DataFrame(recon, index=pivot.index, columns=pivot.columns)
        sim_pblh = mwr_parcel_method(recon_df, offset=0.5)
        sim_pblh_list.append(sim_pblh)


    path=r'data\cloudnet-lidar-ceilometer\20260816_cabauw_chm15k_c07f533c.nc'
    netcdf=gp.read_netcdf(path)
    data=netcdf.data

    # Pivot to time × height grid
    vdop = data.pivot(index="time", columns="height", values="beta_smooth")
    # Filter heights up to 5000 m
    vdop_5km = vdop.loc[:, vdop.columns <= 1600]

    times = pd.to_datetime(vdop_5km.index)
    heights = vdop_5km.columns.values
# -----------------------------------------
# One combined plot: Ceilometer + PBLH
# -----------------------------------------

fig, ax = plt.subplots(figsize=(14, 8))

# ============================================================
# Ceilometer β_raw (0–1600 m)
# ============================================================

path = r'data\cloudnet-lidar-ceilometer\20260816_cabauw_chm15k_c07f533c.nc'
netcdf = gp.read_netcdf(path)
data = netcdf.data

vdop = data.pivot(index="time", columns="height", values="beta_smooth")
vdop_1600 = vdop.loc[:, vdop.columns <= 1600]

times = pd.to_datetime(vdop_1600.index)
heights = vdop_1600.columns.values

pcm = ax.pcolormesh(
    times,
    heights,
    vdop_1600.T,
    shading="auto",
    cmap="RdBu_r",
    vmin=0,
    vmax=1e-6
)

# Colorbar
cbar = fig.colorbar(pcm, ax=ax, label="β_raw [sr⁻¹ m⁻¹]")

# ============================================================
# Monte‑Carlo PBLH diagnostic (median + 95% range)
# ============================================================

# Build ensemble matrix
sim_matrix = np.vstack([
    sim["pbl_height_parcel"].values for sim in sim_pblh_list
]).T  # shape (T, M)

# Stats
sim_median = np.median(sim_matrix, axis=1)
sim_q025   = np.percentile(sim_matrix, 2.5, axis=1)
sim_q975   = np.percentile(sim_matrix, 97.5, axis=1)

time = pivot.index
obs_pblh = mwr_parcel_method(pivot, offset=0.5)

# Plot ensemble members (light gray)
for sim in sim_pblh_list:
    ax.plot(sim["time"], sim["pbl_height_parcel"],
            color="gray", alpha=0.25, linewidth=1)

# Median
ax.plot(time, sim_median, color="green", linewidth=2, label="MC Median")

# 95% band
ax.fill_between(time, sim_q025, sim_q975,
                color="green", alpha=0.2, label="MC 95% Range")

# Observed PBLH
ax.plot(obs_pblh["time"], obs_pblh["pbl_height_parcel"],
        color="red", linewidth=2, label="Observed")

# ============================================================
# Final styling
# ============================================================

ax.set_ylabel("Height [m]")
ax.set_xlabel("Time")
ax.set_title("Ceilometer β_raw + MWR Parcel‑Method PBLH (Monte‑Carlo)")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()
