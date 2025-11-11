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