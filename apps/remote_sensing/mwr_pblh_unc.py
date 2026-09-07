import gruanpy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mwr_pblh import mwr_pre_proc, mwr_parcel_method
from mwr_pca_ssm import apply_pca, pca_diagnostic, reconstruct_observation, reconstruction_diagnostic, pblh_diagnostic
from gruanpy.ssm.statsmodels.multivariate import MultivariateLLL

def simulate_ssm(model, M, seed=42):
    simulator=model.simulation_smoother(seed=seed) # default method is KFS; (method='cfa')  # can specify CFA method
    simulations=[]
    for _ in range(M):
        simulator.simulate()
        simulated_state=simulator.simulated_state.T
        simulations.append(simulated_state)
    return simulations

def reconstruct_simulations(scores, simulations, pca, means, stdevs, n_components):
    """
    Reconstructs potential temperature fields from simulated PCA state trajectories.
    """

    reconstructed_list = []

    for sim_state in simulations:
        # Start from original scores
        scores_smooth = scores.copy()

        # Replace the first n_components with the simulated state
        # assuming sim_state has shape (T, n_components)
        scores_smooth[:, :n_components] = sim_state[:, :n_components]

        # Reconstruct standardized field
        recon_std = scores_smooth @ pca.components_

        # Undo standardization
        reconstructed = recon_std * stdevs + means

        reconstructed_list.append(reconstructed)

    return reconstructed_list

def pblh_monte_carlo_diagnostic(
    pivot,
    obs_pblh,
    sim_pblh_list,
    title="Monte Carlo PBLH Diagnostic"
):
    """
    Monte Carlo PBLH diagnostic using median and 95% range (2.5–97.5%).
    """

    time = pivot.index

    # ---------------------------------------------------------
    # Collect ensemble into matrix (T, M)
    # ---------------------------------------------------------
    sim_matrix = np.vstack([
        sim["pbl_height_parcel"].values for sim in sim_pblh_list
    ])  # shape (M, T)
    sim_matrix = sim_matrix.T          # shape (T, M)

    # Ensemble statistics (median + 95% range)
    sim_median = np.median(sim_matrix, axis=1)
    sim_q025   = np.percentile(sim_matrix, 2.5, axis=1)
    sim_q975   = np.percentile(sim_matrix, 97.5, axis=1)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    # Monte Carlo members
    for sim in sim_pblh_list:
        ax.plot(
            sim["time"],
            sim["pbl_height_parcel"],
            color="gray",
            alpha=0.3,
            linewidth=1
        )

    # Median curve
    ax.plot(
        time,
        sim_median,
        color="green",
        linewidth=2,
        label="MC Median"
    )

    # 95% range band (2.5–97.5 percentile)
    ax.fill_between(
        time,
        sim_q025,
        sim_q975,
        color="green",
        alpha=0.2,
        label="MC 95% Range (2.5–97.5%)"
    )

    # Observed PBLH
    ax.plot(
        obs_pblh["time"],
        obs_pblh["pbl_height_parcel"],
        color="red",
        linewidth=2,
        label="Observed"
    )

    ax.set_title(title)
    ax.set_ylabel("PBL Height [m]")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

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

        
    # -----------------------------------------
    # Monte Carlo PBLH diagnostic
    # -----------------------------------------
    pblh_monte_carlo_diagnostic(
    pivot=pivot,
    obs_pblh=mwr_parcel_method(pivot, offset=0.5),
    sim_pblh_list=sim_pblh_list,
    title="Monte Carlo Parcel-Method PBLH Diagnostic"
)


