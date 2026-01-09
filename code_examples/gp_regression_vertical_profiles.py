import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
path = r'gdp\icm16\LIN-RS-01_2_RS41-GDP_001_20170303T120000_1-004-002.nc'
gdp = gp.read(path)
gdp.data = gdp.data.head(100)

time = gdp.data['time'].values

seconds = (time - time[0]) / np.timedelta64(1, 's')
seconds = seconds.astype(float)

temp = gdp.data['temp'].values
temp_uc = gdp.data['temp_uc'].values
sigma_temp = (temp_uc / 2)**2   # variance estimate

# --- Prepare GP inputs ---
X = seconds.reshape(-1, 1)
y = temp

# --- Define squared exponential (RBF) kernel ---
kernel = 1.0 * RBF(length_scale=10.0) + WhiteKernel(noise_level=np.mean(sigma_temp))

gp_model = GaussianProcessRegressor(
    kernel=kernel,
    alpha=sigma_temp,     # per‑point noise variance
    normalize_y=True
)

# --- Fit GP ---
gp_model.fit(X, y)

# --- Predict on dense grid ---
X_pred = np.linspace(seconds.min(), seconds.max(), 500).reshape(-1, 1)
y_pred, y_std = gp_model.predict(X_pred, return_std=True)

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(seconds, temp, 'o', markersize=4, color='black',label='Data')
plt.plot(X_pred, y_pred, 'r-', label='GP mean')
plt.fill_between(
    X_pred.ravel(),
    y_pred - 2*y_std,
    y_pred + 2*y_std,
    color='r',
    alpha=0.2,
    label='95% confidence interval'
)

plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (K)')
plt.title('Gaussian Process Fit (Squared Exponential Kernel)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("Optimized kernel:", gp_model.kernel_)

# --- Simulate realizations from the GP posterior ---
n_realizations = 5
y_samples = gp_model.sample_y(X_pred, n_samples=n_realizations, random_state=42)

# --- Plot realizations ---
plt.figure(figsize=(12, 6))
plt.plot(seconds, temp, 'o', markersize=4, label='Data', color='black', zorder=5)
#plt.plot(X_pred, y_pred, 'r-', linewidth=2, label='GP mean')
for i in range(n_realizations):
    plt.plot(X_pred, y_samples[:, i], alpha=0.5, linewidth=1, label=f'Realization {i+1}')
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (K)')
plt.title('GP Realizations from Posterior')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()