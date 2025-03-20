import numpy as np
from scipy import signal

class NoiseGenerator:
    """
    Utility class for generating realistic measurement noise to simulate real-world data collection.
    Includes various noise models relevant to neuronal recordings.
    """
    
    @staticmethod
    def add_measurement_noise(data, noise_level=0.05, noise_type='gaussian'):
       
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
    
    @staticmethod
    def add_pink_noise(data, noise_level=0.05):
    
        if data is None or len(data) == 0:
            return data
            
        # Make a copy to avoid modifying the original data
        noisy_data = data.copy()
        
        # Calculate the data range to scale the noise appropriately
        data_range = np.max(data) - np.min(data)
        
        # Generate pink noise for each dimension/neuron
        if len(data.shape) > 1:
            # For 2D data (multiple neurons)
            for i in range(data.shape[1]):
                # Generate white noise
                white_noise = np.random.normal(0, 1, size=len(data))
                
                # Convert to pink noise using FFT
                # Get the FFT of the white noise
                fft = np.fft.rfft(white_noise)
                
                # Generate 1/f spectrum
                f = np.fft.rfftfreq(len(white_noise))
                f[0] = 1  # Avoid division by zero
                fft = fft / np.sqrt(f)
                
                # Convert back to time domain
                pink = np.fft.irfft(fft)
                
                # Scale and add to data
                pink = pink / np.std(pink) * noise_level * data_range
                noisy_data[:, i] += pink[:len(data)]
        else:
            # For 1D data
            white_noise = np.random.normal(0, 1, size=len(data))
            fft = np.fft.rfft(white_noise)
            f = np.fft.rfftfreq(len(white_noise))
            f[0] = 1  # Avoid division by zero
            fft = fft / np.sqrt(f)
            pink = np.fft.irfft(fft)
            pink = pink / np.std(pink) * noise_level * data_range
            noisy_data += pink[:len(data)]
        
        return noisy_data
    
    @staticmethod
    def add_synaptic_noise(data, noise_level=0.03, tau=10):
        
        if data is None or len(data) == 0:
            return data
            
        # Make a copy to avoid modifying the original data
        noisy_data = data.copy()
        
        # Calculate the data range to scale the noise appropriately
        data_range = np.max(data) - np.min(data)
        sigma = noise_level * data_range
        
        # Generate Ornstein-Uhlenbeck process
        dt = 1.0  # Assuming unit time steps
        theta = 1.0 / tau  # Inverse of time constant
        
        if len(data.shape) > 1:
            # For 2D data (multiple neurons)
            for i in range(data.shape[1]):
                x = 0.0
                noise = np.zeros(len(data))
                for t in range(len(data)):
                    x = x + theta * (0 - x) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
                    noise[t] = x
                noisy_data[:, i] += noise
        else:
            # For 1D data
            x = 0.0
            noise = np.zeros(len(data))
            for t in range(len(data)):
                x = x + theta * (0 - x) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
                noise[t] = x
            noisy_data += noise
        
        return noisy_data
    
    @staticmethod
    def add_periodic_artifact(data, amplitude=0.04, frequency=0.1, phase=0):
      
        if data is None or len(data) == 0:
            return data
            
        # Make a copy to avoid modifying the original data
        artifact_data = data.copy()
        
        # Calculate the data range to scale the artifact appropriately
        data_range = np.max(data) - np.min(data)
        artifact_amplitude = amplitude * data_range
        
        # Generate time points
        time_points = np.arange(len(data))
        
        # Generate periodic signal
        artifact = artifact_amplitude * np.sin(2 * np.pi * frequency * time_points + phase)
        
        # Apply artifact
        if len(data.shape) > 1:
            # For 2D data (multiple neurons)
            for i in range(data.shape[1]):
                # Add slightly different phase for each neuron to make it more realistic
                neuron_phase = phase + 0.2 * i
                artifact = artifact_amplitude * np.sin(2 * np.pi * frequency * time_points + neuron_phase)
                artifact_data[:, i] += artifact
        else:
            # For 1D data
            artifact_data += artifact
        
        return artifact_data