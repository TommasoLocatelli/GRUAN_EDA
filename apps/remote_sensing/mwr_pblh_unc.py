import gruanpy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from scipy.ndimage import uniform_filter1d

# ============================================================
# 1. Load Cloudnet MWR product
# ============================================================

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
data["time"] = pd.to_datetime(data["time"])
# ============================================================
# 2. Build PCA + State-Space Model
# ============================================================

pt = data.pivot(index="time", columns="height", values="potential_temperature")
pt_clean = pt.dropna(axis=1, how='any')

k_factors = 2
N_COMPONENTS = k_factors   # MUST MATCH

pca = PCA(n_components=N_COMPONENTS)
scores = pca.fit_transform(pt_clean)

mod = DynamicFactor(scores, k_factors=k_factors, factor_order=2)
res = mod.fit(maxiter=500)

k_states = mod.k_states
factor_order = mod.factor_order


# ============================================================
# 3. Simulation smoother
# ============================================================

def simulate_ssm(model, M, seed=42):
    simulator = model.simulation_smoother(seed=seed)
    simulations = []
    for _ in range(M):
        simulator.simulate()
        simulations.append(simulator.simulated_state)
    return simulations

N_MC = 200
sim_states = simulate_ssm(mod, N_MC)

# ============================================================
# 4. Reconstruct θ fields from simulated states
# ============================================================

def reconstruct_theta(sim_state, pca, k_factors, factor_order):
    # sim_state shape: (k_states, T)
    T = sim_state.shape[1]

    k_factor_states = k_factors * factor_order
    factor_states = sim_state[:k_factor_states, :]  # (k_factor_states, T)

    # take first k_factors rows as scores
    scores_sim = factor_states[:k_factors, :].T     # (T, k_factors)

    return scores_sim @ pca.components_ + pca.mean_

# ============================================================
# 5. Inject reconstructed θ into original dataframe
# ============================================================

def inject_theta(data, pt_smooth):
    df = data.copy()
    df = df.set_index(["time", "height"])
    for t in pt_smooth.index:
        for h in pt_smooth.columns:
            if (t, h) in df.index:
                df.loc[(t, h), "potential_temperature"] = pt_smooth.loc[t, h]
    return df.reset_index()

# ============================================================
# 6. PBLH retrieval (RH-gradient, θ-gradient, modified parcel)
# ============================================================

def compute_pblh(data, dtheta=0.5, n_consecutive=3):
    # RH gradient
    grid_rh = data.pivot(index="height", columns="time", values="relative_humidity")
    height_vals = grid_rh.index.values
    rh_grad = np.gradient(grid_rh.values, height_vals, axis=0)
    rh_grad_df = pd.DataFrame(rh_grad, index=grid_rh.index, columns=grid_rh.columns)
    rh_grad_long = rh_grad_df.stack().reset_index()
    rh_grad_long.columns = ["height", "time", "rh_gradient"]
    pblh_rh = rh_grad_long.groupby("time").apply(
        lambda g: g.loc[g["rh_gradient"].idxmin()]["height"]
    )

    # θ gradient
    theta_grid = data.pivot(index="height", columns="time", values="potential_temperature")
    theta_grad = np.gradient(theta_grid.values, height_vals, axis=0)
    theta_grad_df = pd.DataFrame(theta_grad, index=theta_grid.index, columns=theta_grid.columns)
    theta_grad_long = theta_grad_df.stack().reset_index()
    theta_grad_long.columns = ["height", "time", "theta_gradient"]
    pblh_theta = theta_grad_long.groupby("time").apply(
        lambda g: g.loc[g["theta_gradient"].idxmax()]["height"]
    )

    # Modified parcel method (smoothed θ, threshold, consecutive levels)
    heights = theta_grid.index.values
    times = theta_grid.columns
    parcel_mod = []

    for t in times:
        profile = theta_grid[t].values

        # vertical smoothing
        profile_smooth = uniform_filter1d(profile, size=3)

        theta0 = profile_smooth[0]
        mask = profile_smooth > (theta0 + dtheta)

        consec = np.convolve(mask.astype(int),
                             np.ones(n_consecutive, dtype=int),
                             mode='same')

        idx = np.argmax(consec >= n_consecutive)
        if consec[idx] >= n_consecutive:
            parcel_mod.append(heights[idx])
        else:
            parcel_mod.append(np.nan)

    pblh_parcel_mod = pd.Series(parcel_mod, index=times)

    return pblh_rh, pblh_theta, pblh_parcel_mod

# ============================================================
# 7. Monte Carlo loop
# ============================================================

ens_rh = []
ens_theta = []
ens_parcel_mod = []

for sim_state in sim_states:
    pt_smooth = reconstruct_theta(sim_state, pca, k_factors=2, factor_order=2)
    pt_smooth = pd.DataFrame(pt_smooth, index=pt_clean.index, columns=pt_clean.columns)

    data_mc = inject_theta(data, pt_smooth)

    pblh_rh, pblh_theta, pblh_parcel_mod = compute_pblh(data_mc)

    ens_rh.append(pblh_rh)
    ens_theta.append(pblh_theta)
    ens_parcel_mod.append(pblh_parcel_mod)

ens_rh = pd.DataFrame(ens_rh).T
ens_theta = pd.DataFrame(ens_theta).T
ens_parcel_mod = pd.DataFrame(ens_parcel_mod).T

unc_rh = ens_rh.std(axis=1)
unc_theta = ens_theta.std(axis=1)
unc_parcel_mod = ens_parcel_mod.std(axis=1)

# ============================================================
# 8. Plot thermal PBLH uncertainty
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(ens_theta.index, ens_theta.mean(axis=1), color="black", label="Thermal PBLH (mean)")
ax.fill_between(
    ens_theta.index,
    ens_theta.mean(axis=1) - unc_theta,
    ens_theta.mean(axis=1) + unc_theta,
    color="gray", alpha=0.3, label="±1σ"
)

ax.set_title("MWR Thermal PBLH Uncertainty (Monte Carlo SSM)")
ax.set_ylabel("Height [m]")
ax.set_xlabel("Time")
ax.legend()
plt.tight_layout()
plt.show()
