import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from neuronModels.PhysicsModels import PhysicsModels
from utils.Constants import Constants
from utils.Logger import Logger, LogLevel

logger = Logger()


class EnhancedIzhikevichModel(PhysicsModels):
    """
    Enhanced Izhikevich model with improved parameter handling and configurable settings
    for better neuronal dynamics simulation and forecasting.
    """
    N = 20
    A = 0.05
    B = 0.2 
    C = -65
    D = 8 
    I = 10
    THRESHOLD = 30

    def __init__(self, N=Constants.N, a=A, b=B, c=C, d=D, I=I, threshold=THRESHOLD, 
                 use_adaptive_params=False, noise_level=0.0):
        """
        Initialize the enhanced Izhikevich model with configurable parameters.
        
        Parameters:
        -----------
        N : int
            Number of neurons
        a : float
            Time scale of recovery variable u
        b : float
            Sensitivity of recovery variable u to membrane potential v
        c : float
            After-spike reset value of membrane potential v
        d : float
            After-spike reset increment of recovery variable u
        I : float
            External current input
        threshold : float
            Spike threshold
        use_adaptive_params : bool
            Whether to use adaptive parameters for heterogeneous neuron populations
        noise_level : float
            Level of noise to add to the simulation (0.0-1.0)
        """
        self.N = N  # Number of neurons
        
        # Handle adaptive parameters for heterogeneous neuron populations
        if use_adaptive_params:
            # Create parameter distributions for different neuron types
            self.a = np.random.normal(a, 0.01, N)  # Slightly varied a parameters
            self.b = np.random.normal(b, 0.02, N)  # Slightly varied b parameters
            self.c = np.random.normal(c, 2.0, N)   # Varied reset potential
            self.d = np.random.normal(d, 1.0, N)   # Varied reset increment
        else:
            # Use uniform parameters
            self.a = np.ones(N) * a
            self.b = np.ones(N) * b
            self.c = np.ones(N) * c
            self.d = np.ones(N) * d
            
        self.I = I  # External current
        self.threshold = threshold  # Spike threshold
        self.noise_level = noise_level  # Noise level for simulation
        
        logger.log("Instantiated enhanced Izhikevich model", LogLevel.INFO)

    def _model(self, t, y):
        """
        Differential equations for the Izhikevich model.
        
        Parameters:
        -----------
        t : float
            Current time point
        y : numpy.ndarray
            Current state vector [v1, v2, ..., vN, u1, u2, ..., uN]
            
        Returns:
        --------
        numpy.ndarray
            Rate of change for each state variable
        """
        N = len(y) // 2
        v = y[:N]  # Membrane potential
        u = y[N:]  # Recovery variable

        # Add time-dependent or noise-based current if needed
        I_effective = self.I
        if self.noise_level > 0:
            I_effective += np.random.normal(0, self.noise_level * self.I, N)

        # Izhikevich model equations
        dv = 0.04 * v**2 + 5 * v + 140 - u + I_effective
        du = self.a * (self.b * v - u)

        # Spike detection and reset
        spike = v >= self.threshold
        v[spike] = self.c[spike] if isinstance(self.c, np.ndarray) else self.c
        u[spike] += self.d[spike] if isinstance(self.d, np.ndarray) else self.d

        return np.concatenate([dv, du])

    def _generate_synthetic_data(self, T=200, dt=0.1, use_saved_models=True):
        """
        Generate synthetic data using the Izhikevich model.
        
        Parameters:
        -----------
        T : float
            Total simulation time
        dt : float
            Time step for simulation
        use_saved_models : bool
            Whether to load previously saved data
            
        Returns:
        --------
        tuple
            (time_points, state_variables)
        """
        if use_saved_models:
            t_points, v_points = self._load_datas(Constants.IZH)
        else: 
            # Initial conditions with controlled randomization
            v0 = np.random.uniform(-70, -60, self.N)  # Initial membrane potentials
            u0 = self.b * v0  # Initial recovery variables
            y0 = np.concatenate([v0, u0])  # Combined initial state

            # Time evaluation points
            t_eval = np.arange(0, T, dt)

            # Solve the initial value problem with adaptive step size
            sol = solve_ivp(
                self._model,
                [0, T],
                y0,
                method='RK45',  # Runge-Kutta 4(5) method
                t_eval=t_eval,
                vectorized=True,
                rtol=1e-6,  # Relative tolerance
                atol=1e-9   # Absolute tolerance
            )
            
            t_points = sol.t
            v_points = sol.y.T
            
            logger.log("Generated synthetic data for enhanced Izhikevich model", LogLevel.INFO)
            self._save_datas(Constants.IZH, t_points, v_points)

        return t_points, v_points


class IzhikevichNetworkModel(nn.Module):
    """
    Neural network model for forecasting Izhikevich neuron dynamics using
    a combination of recurrent and feed-forward layers.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2, use_batch_norm=True):
        """
        Initialize the Izhikevich network model.
        
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
        super(IzhikevichNetworkModel, self).__init__()
        
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