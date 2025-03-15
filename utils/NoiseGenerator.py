import numpy as np

class NoiseGenerator:
    """
    Utility class for generating realistic measurement noise to simulate real-world data collection.
    """
    
    @staticmethod
    def add_measurement_noise(data, noise_level=0.05, noise_type='gaussian'):
        """
        Add realistic measurement noise to synthetic data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            The synthetic data to which noise will be added
        noise_level : float
            The standard deviation of the noise relative to the data range
        noise_type : str
            The type of noise to add ('gaussian', 'uniform', or 'spike')
            
        Returns:
        --------
        numpy.ndarray
            The data with added noise
        """
        if data is None or len(data) == 0:
            return data
            
        # Make a copy to avoid modifying the original data
        noisy_data = data.copy()
        
        # Calculate the data range to scale the noise appropriately
        data_range = np.max(data) - np.min(data)
        noise_std = noise_level * data_range
        
        if noise_type == 'gaussian':
            # Add Gaussian noise (most common in real measurements)
            noise = np.random.normal(0, noise_std, size=data.shape)
            noisy_data = data + noise
            
        elif noise_type == 'uniform':
            # Add uniform noise
            noise = np.random.uniform(-noise_std*1.7, noise_std*1.7, size=data.shape)
            noisy_data = data + noise
            
        elif noise_type == 'spike':
            # Add occasional measurement spikes (simulates sensor glitches)
            noise = np.random.normal(0, noise_std*0.5, size=data.shape)
            # Add random spikes (about 1% of measurements)
            spike_mask = np.random.random(size=data.shape) < 0.01
            noise[spike_mask] = np.random.normal(0, noise_std*5, size=np.sum(spike_mask))
            noisy_data = data + noise
        
        return noisy_data
    
    @staticmethod
    def add_systematic_error(data, bias=0.02, drift_factor=0.001):
        """
        Add systematic errors like bias and drift to synthetic data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            The synthetic data to which systematic errors will be added
        bias : float
            Constant offset as a fraction of data range
        drift_factor : float
            Rate of linear drift over time as a fraction of data range
            
        Returns:
        --------
        numpy.ndarray
            The data with added systematic errors
        """
        if data is None or len(data) == 0:
            return data
            
        # Make a copy to avoid modifying the original data
        error_data = data.copy()
        
        # Calculate the data range to scale the errors appropriately
        data_range = np.max(data) - np.min(data)
        
        # Add constant bias
        bias_value = bias * data_range
        
        # Add time-dependent drift
        time_points = np.arange(len(data))
        drift = drift_factor * data_range * time_points / len(data)
        
        # Apply both error types
        if len(data.shape) > 1:
            # For 2D data (multiple neurons)
            for i in range(data.shape[1]):
                error_data[:, i] += bias_value + drift
        else:
            # For 1D data
            error_data += bias_value + drift
        
        return error_data