import numpy as np
import torch
import torch.nn as nn

from neuronModels.PhysicsModels import PhysicsModels
from utils.Constants import Constants
from utils.Logger import Logger, LogLevel

logger = Logger()


class EnhancedLifModel(PhysicsModels):
    """
    Enhanced Leaky Integrate-and-Fire (LIF) model with improved parameter handling 
    and configurable settings for better neuronal dynamics simulation and forecasting.
    """
    TAU = 10.0        # Membrane time constant (ms)
    R = 5.0           # Membrane resistance (MΩ)
    I = 10.0          # Input current (pA)
    V_REST = -65      # Resting potential (mV)
    V_TH = -55        # Threshold potential (mV)
    V_RESET = -75     # Reset potential (mV)
    REFRACTORY = 2.0  # Refractory period (ms)

    def __init__(self, N=Constants.N, tau=TAU, R=R, I=I, V_rest=V_REST, V_th=V_TH, V_reset=V_RESET,
                 refractory=REFRACTORY, use_adaptive_params=False, noise_level=0.0):
        """
        Initialize the enhanced LIF model with configurable parameters.
        
        Parameters:
        -----------
        N : int
            Number of neurons
        tau : float
            Membrane time constant (ms)
        R : float
            Membrane resistance (MΩ)
        I : float or callable
            Input current (pA) or a function that returns current given time
        V_rest : float
            Resting membrane potential (mV)
        V_th : float
            Threshold potential (mV)
        V_reset : float
            Reset potential after spike (mV)
        refractory : float
            Refractory period after spike (ms)
        use_adaptive_params : bool
            Whether to use adaptive parameters for heterogeneous neuron populations
        noise_level : float
            Level of noise to add to the simulation (0.0-1.0)
        """
        self.N = N  # Number of neurons
        
        # Handle adaptive parameters for heterogeneous neuron populations
        if use_adaptive_params:
            # Create parameter distributions for different neuron types
            self.tau = np.random.normal(tau, tau * 0.1, N)  # Varied time constants
            self.R = np.random.normal(R, R * 0.05, N)  # Varied resistances
            self.V_rest = np.random.normal(V_rest, 2.0, N)  # Varied resting potentials
            self.V_th = np.random.normal(V_th, 1.0, N)  # Varied threshold potentials
            self.V_reset = np.random.normal(V_reset, 2.0, N)  # Varied reset potentials
            self.refractory = np.random.normal(refractory, 0.5, N)  # Varied refractory periods
        else:
            # Use uniform parameters
            self.tau = np.ones(N) * tau
            self.R = np.ones(N) * R
            self.V_rest = np.ones(N) * V_rest
            self.V_th = np.ones(N) * V_th
            self.V_reset = np.ones(N) * V_reset
            self.refractory = np.ones(N) * refractory
        
        self.I = I  # Input current
        self.noise_level = noise_level  # Noise level for simulation
        self.spikes = [[] for _ in range(N)]  # Spike times for each neuron
        self.last_spike = np.full(N, -np.inf)  # Last spike time for each neuron
        
        logger.log("Instantiated enhanced LIF model", LogLevel.INFO)

    def _model(self, t, V):
        """
        Differential equation for the LIF model.
        
        Parameters:
        -----------
        t : float
            Current time point
        V : numpy.ndarray
            Current membrane potentials
            
        Returns:
        --------
        numpy.ndarray
            Rate of change for membrane potentials
        """
        # Calculate effective current (with optional time-dependence and noise)
        I_effective = np.zeros(self.N)
        for i in range(self.N):
            # Get base current (time-dependent or constant)
            if callable(self.I):
                I_effective[i] = self.I(t)
            else:
                I_effective[i] = self.I
                
            # Add noise if specified
            if self.noise_level > 0:
                I_effective[i] += np.random.normal(0, self.noise_level * abs(self.I), 1)[0]
        
        # Calculate membrane potential changes
        dVdt = np.zeros(self.N)
        for i in range(self.N):
            # Check if neuron is in refractory period
            if t - self.last_spike[i] > self.refractory[i]:
                # LIF equation: dV/dt = (-(V - V_rest) + R*I) / tau
                dVdt[i] = (-(V[i] - self.V_rest[i]) + self.R[i] * I_effective[i]) / self.tau[i]
        
        return dVdt

    def _generate_synthetic_data(self, T=200, dt=0.1, use_saved_models=True):
        """
        Generate synthetic data using the LIF model.
        
        Parameters:
        -----------
        T : float
            Total simulation time (ms)
        dt : float
            Time step for simulation (ms)
        use_saved_models : bool
            Whether to load previously saved data
            
        Returns:
        --------
        tuple
            (time_points, membrane_potentials)
        """
        if use_saved_models:
            t_points, v_points = self._load_datas(Constants.LIF)
        else:
            # Initialize simulation variables
            t_points = np.arange(0, T, dt)
            v_points = np.full((len(t_points), self.N), self.V_rest)  # Initialize with resting potentials
            V_current = np.array([self.V_rest[i] for i in range(self.N)])  # Current membrane potentials
            
            # Reset spike tracking
            self.spikes = [[] for _ in range(self.N)]
            self.last_spike = np.full(self.N, -np.inf)
            
            # Run simulation with Euler method
            for i, t in enumerate(t_points[:-1]):
                # Calculate next voltage using the model
                dV = self._model(t, V_current) * dt
                V_next = V_current + dV
                
                # Check for spikes and apply reset
                for j in range(self.N):
                    if V_next[j] >= self.V_th[j]:
                        # Record spike
                        self.spikes[j].append(t)
                        self.last_spike[j] = t
                        # Reset membrane potential
                        V_next[j] = self.V_reset[j]
                
                # Store results and update current state
                v_points[i + 1] = V_next
                V_current = V_next
            
            logger.log("Generated synthetic data for enhanced LIF model", LogLevel.INFO)
            self._save_datas(Constants.LIF, t_points, v_points)

        return t_points, v_points


class LifNetworkModel(nn.Module):
    """
    Neural network model for forecasting LIF neuron dynamics using
    a combination of recurrent and feed-forward layers.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2, use_batch_norm=True):
        """
        Initialize the LIF network model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of hidden units in each layer
        output_size : int
            Number of output features
        num_layers : int
            Number of recurrent layers
        dropout : float
            Dropout probability (0-1)
        use_batch_norm : bool
            Whether to use batch normalization
        """
        super(LifNetworkModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        # GRU layers for sequence processing
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size) if use_batch_norm else None
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size)
        """
        # GRU layers
        gru_out, _ = self.gru(x)
        
        # Take the output from the last time step
        out = gru_out[:, -1, :]
        
        # Apply batch normalization if enabled
        if self.use_batch_norm and self.batch_norm is not None:
            out = self.batch_norm(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out
    
    def predict(self, x):
        """
        Make predictions in evaluation mode.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        torch.Tensor
            Predicted output
        """
        self.eval()
        with torch.no_grad():
            return self(x)