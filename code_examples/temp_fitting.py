import sys
import os
import matplotlib.pyplot as plt
import numpy.random as random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gruanpy import gruanpy as gp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

folder = r'gdp\products_RS41-GDP-1_POT_2025'
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:5]:
    gdp = gp.read(file_path)
    gdp.data = gdp.data[gdp.data['alt'] <= 2000]  # Limit to first 10 km for speed
    where = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]
    when = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
    
    alt = gdp.data['alt'].values.reshape(-1, 1)
    temp = gdp.data['temp'].values

    kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        noise_level=1, noise_level_bounds=(1e-10, 1e1)
    )
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

    n_samples = len(alt)-10
    indices = np.random.choice(len(alt), size=n_samples, replace=False)

    alt_sub = alt[indices]
    temp_sub = temp[indices]

    # Fit the GPR model on the subsample
    gpr = GaussianProcessRegressor()
    gpr.fit(alt_sub, temp_sub)

    # Predict on the full alt range or another set
    y_mean, y_std = gpr.predict(alt, return_std=True)


    plt.scatter(temp, alt, label='Observed')
    plt.plot(y_mean, alt, color='red', label='Fit')
    plt.xlabel('Altitude (m)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Profile at {where} on {when}')
    plt.legend()
    plt.show()
    break



