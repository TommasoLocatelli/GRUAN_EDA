import numpy as np

class GeostatisticMethods:
    """
    A class that contains methods for geostatistical analysis.
    """

    def variogram(self, data, d, max_distance, min_distance=1, num_bins=100):
        """
        Calculate the empirical variogram of the input data.

        Parameters:
        data (array-like): The input spatial data points.
        d (int): The dimension of the spatial data (e.g., 2 for 2D, 3 for 3D).
        max_distance (float): The maximum distance to consider for the variogram.
        num_bins (int): The number of bins to divide the distance range into.

        Returns:
        tuple: A tuple containing the bin centers and the corresponding variogram values.
        """
        from scipy.spatial.distance import pdist, squareform

        # Calculate pairwise distances
        distances = pdist(data[:, :d])
        values = pdist(data[:, d].reshape(-1, 1))

        # Create bins
        bin_edges = np.linspace(min_distance, max_distance, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        variogram_values = np.zeros(num_bins)

        # Calculate variogram values for each bin
        for i in range(num_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            if np.any(mask):
                variogram_values[i] = 0.5 * np.mean(values[mask] ** 2)
            else:
                variogram_values[i] = np.nan

        return bin_centers, variogram_values