from sklearn.kernel_ridge import KernelRidge
import numpy as np
import matplotlib.pyplot as plt
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
krr = KernelRidge(alpha=1.0)
krr.fit(X, y)
y_pred = krr.predict(X)
plt.scatter(range(n_samples), y, label='True values')
plt.scatter(range(n_samples), y_pred, label='Predicted values')
plt.legend()
plt.show()
