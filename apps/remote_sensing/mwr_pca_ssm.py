import gruanpy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mwr_pblh import mwr_pre_proc, mwr_parcel_method
from sklearn.decomposition import PCA
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL

def apply_pca(pivot, n_components):
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(pivot)

    return pca, scores

def pca_diagnostic(pca, pivot, X_std):
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())

    # Bar plot of explained variance
    labels = [f"PC{i+1}" for i in range(N_COMPONENTS)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, pca.explained_variance_ratio_)
    ax.set_title("PCA Explained Variance Ratio")
    ax.set_ylabel("Fraction of variance")
    plt.show()

    # Plot PCA vertical modes (loadings)
    fig, axes = plt.subplots(N_COMPONENTS, 1, figsize=(10, 4*N_COMPONENTS), sharex=True)

    for i in range(N_COMPONENTS):
        axes[i].plot(pivot.columns, pca.components_[i, :])
        axes[i].set_title(f"PCA Mode {i+1}")
        axes[i].set_ylabel("Loading")

    axes[-1].set_xlabel("Height [m]")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot PCA scores (time series)
    # -----------------------------
    scores = pca.transform(X_std)

    fig, axes = plt.subplots(N_COMPONENTS, 1, figsize=(12, 3*N_COMPONENTS), sharex=True)

    for i in range(N_COMPONENTS):
        axes[i].plot(scores[:, i])
        axes[i].set_title(f"PCA Score Series – PC{i+1}")
        axes[i].set_ylabel("Score")

    axes[-1].set_xlabel("Time index")
    plt.tight_layout()
    plt.show()

def reconstruct_observation(scores, results, n_components, means, stdevs):

    scores_smooth = scores.copy()
    scores_smooth[:, :n_components] = results.fittedvalues[:, :n_components]

    # Reconstruct standardized field
    recon_std = scores_smooth @ pca.components_

    # Undo standardization
    reconstructed = recon_std * stdevs + means

    return reconstructed

def reconstruction_diagnostic(pivot, reconstructed):
    diff = pivot.values - reconstructed

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

    vmin = pivot.values.min()
    vmax = pivot.values.max()
    im0 = axes[0].pcolormesh(pivot.index, pivot.columns, pivot.values.T,
                            shading='auto', cmap='viridis',
                            vmin=vmin, vmax=vmax)

    im1 = axes[1].pcolormesh(pivot.index, pivot.columns, reconstructed.T,
                            shading='auto', cmap='viridis',
                            vmin=vmin, vmax=vmax)

    im2 = axes[2].pcolormesh(pivot.index, pivot.columns, diff.T,
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

def pblh_diagnostic(pivot, reconstructed_df, obs_pblh, smooth_pblh):
    """
    Three-panel diagnostic:
    1. Original PT + Observed PBLH
    2. Reconstructed PT + Smoothed PBLH
    3. Difference field
    """

    diff = pivot.values - reconstructed_df.values

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

    # ---------------------------------------------------------
    # 1. Original PT + Observed PBLH
    # ---------------------------------------------------------
    vmin = pivot.values.min()
    vmax = pivot.values.max()

    im0 = axes[0].pcolormesh(
        pivot.index, pivot.columns, pivot.values.T,
        shading='auto', cmap='viridis', vmin=vmin, vmax=vmax
    )

    axes[0].plot(
        obs_pblh["time"], obs_pblh["pbl_height_parcel"],
        color="red", linewidth=2, label="Observed PBLH"
    )

    axes[0].plot(#********
        smooth_pblh["time"], smooth_pblh["pbl_height_parcel"],
        color="orange", linewidth=2, label="Smoothed PBLH"
    )

    axes[0].set_title("Original Potential Temperature + Observed PBLH")
    axes[0].set_ylabel("Height [m]")
    axes[0].legend()
    fig.colorbar(im0, ax=axes[0], label="K")

    # ---------------------------------------------------------
    # 2. Reconstructed PT + Smoothed PBLH
    # ---------------------------------------------------------
    im1 = axes[1].pcolormesh(
        pivot.index, pivot.columns, reconstructed_df.values.T,
        shading='auto', cmap='viridis', vmin=vmin, vmax=vmax
    )

    axes[1].plot(
        smooth_pblh["time"], smooth_pblh["pbl_height_parcel"],
        color="orange", linewidth=2, label="Smoothed PBLH"
    )

    axes[1].set_title("Reconstructed (State-Space Smoothed) + Smoothed PBLH")
    axes[1].set_ylabel("Height [m]")
    axes[1].legend()
    fig.colorbar(im1, ax=axes[1], label="K")

    # ---------------------------------------------------------
    # 3. Difference field
    # ---------------------------------------------------------
    im2 = axes[2].pcolormesh(
        pivot.index, pivot.columns, diff.T,
        shading='auto', cmap='coolwarm'
    )

    axes[2].set_title("Difference (Original - Reconstructed)")
    axes[2].set_ylabel("Height [m]")
    axes[2].set_xlabel("Time")
    fig.colorbar(im2, ax=axes[2], label="K")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
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

    data = mwr_pre_proc(netcdf)

    pivot = data.pivot(index="time", columns="height", values="potential_temperature").dropna(axis=1, how='any')

    N_COMPONENTS=7

    means = pivot.values.mean(axis=0)
    stdevs = pivot.values.std(axis=0, ddof=0)
    X_std = (pivot.values - means) / stdevs
    pca, scores = apply_pca(X_std, N_COMPONENTS)

    pca_diagnostic(pca, pivot, X_std)

    ssm=MultivariateLLL(scores)
    results=ssm.fit(maxiter=500)

    print(results.summary())

    reconstructed=reconstruct_observation(scores, results, N_COMPONENTS, means, stdevs)

    reconstruction_diagnostic(pivot, reconstructed)

    obs_pblh=mwr_parcel_method(pivot, offset=0.5)
    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=pivot.index,
        columns=pivot.columns
    )

    smooth_pblh=mwr_parcel_method(reconstructed_df, offset=0.5)

    pblh_diagnostic(pivot, reconstructed_df, obs_pblh, smooth_pblh)
