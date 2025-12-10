import numpy.random as random
import numpy as np

class NoiseMethods:
    """
    A class that contains methods for adding noise to data.
    """
    import numpy as np

    def gaussian_noise(self, data=0, std=1):
        """
        Add Gaussian noise to the input data.

        Parameters:
        data (array-like): The input data to which noise will be added.
        std (float or array-like): The standard deviation of the Gaussian noise.
        Returns:
        array-like: The input data with added Gaussian noise.
        """
        data = np.asarray(data)
        std = np.asarray(std)
        return np.random.normal(loc=data, scale=std, size=data.shape)
    
    def smooth_noise(self, data=0, std=1, window_size=10):
        """
        Add smoothed Gaussian noise to the input data.

        Parameters:
        data (array-like): The input data to which noise will be added.
        std (float or array-like): The standard deviation of the Gaussian noise.
        window_size (int): The size of the smoothing window.

        Returns:
        array-like: The input data with added smoothed Gaussian noise.
        """
        data = np.asarray(data)
        std = np.asarray(std)
        noise = np.random.normal(loc=0, scale=std, size=data.shape)
        smoothed_noise = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
        return data + smoothed_noise
    
    def uncertainty_matrix(self, data, variable_clmn='temp', var_unc='_uc', var_unc_corr='_uc_scor', k=2):
        """
        Create a covariance/uncertainty matrix for a given variable
        considering both uncorrelated and correlated uncertainties.
        Squared uncorrelated uncertainties are placed on the diagonal,
        while correlated uncertainties product are placed on the off-diagonal elements.
        """
        n = len(data)
        cov_matrix = np.zeros((n, n))
        unc = data[variable_clmn+var_unc].values
        correlated_unc = data[variable_clmn+var_unc_corr].values

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = (unc[i]/k) ** 2
                else:
                    cov_matrix[i, j] = (correlated_unc[i]/k) * (correlated_unc[j]/k)
        return cov_matrix    
    
    def gp_noise(self, data, uncertainty_matrix, method='cholesky'):
        """
        Generate correlated noise using Gaussian Process methods.

        Parameters:
        data (array-like): The input data to which noise will be added.
        uncertainty_matrix (2D array-like): The covariance/uncertainty matrix.
        method (str): The method to use for generating correlated noise. Options are 'cholesky' or 'svd'.

        Returns:
        array-like: The generated correlated noise.
        """
        if method == 'cholesky':
            L = np.linalg.cholesky(uncertainty_matrix)
            N = len(data)
            standard_normal_samples = np.random.normal(0, 1, N)  # generate standard normal samples
            noise = L @ standard_normal_samples  # generate correlated noise
        elif method == 'svd':
            U, S, Vt = np.linalg.svd(uncertainty_matrix)
            A = U @ np.sqrt(np.diag(S))  # get matrix A from SVD
            N = len(data)
            standard_normal_samples = np.random.normal(0, 1, N)  # generate standard normal samples
            noise = A @ standard_normal_samples  # generate correlated noise
        return noise