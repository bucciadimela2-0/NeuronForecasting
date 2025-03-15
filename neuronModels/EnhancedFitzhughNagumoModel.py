import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from neuronModels.PhysicsModels import PhysicsModels
from utils.Constants import Constants
from utils.DataHandler import DataHandler
from utils.Logger import Logger, LogLevel

logger = Logger()


class EnhancedFitzhughNagumoModel(PhysicsModels):
    """
    Enhanced Fitzhugh-Nagumo model with improved parameter handling and configurable settings
    for better neuronal dynamics simulation and forecasting.
    """
    EPSILON = 0.05
    SIGMA = 0.1  # Coupling strength
    A = 0.5  # Threshold parameter
    PHI = np.pi / 2 + 0.1
    B = np.array([[np.cos(PHI), np.sin(PHI)], [-np.sin(PHI), np.cos(PHI)]])
    G = DataHandler.generate_adjacency_matrix(Constants.A1, n=3)

    def __init__(self, N=Constants.N, epsilon=EPSILON, sigma=SIGMA, a=A, B=B, G=G,
                 use_adaptive_params=False, noise_level=0.0):
        """
        Initialize the enhanced Fitzhugh-Nagumo model with configurable parameters.
        
        Parameters:
        -----------
        N : int
            Number of neurons
        epsilon : float
            Time scale parameter
        sigma : float
            Coupling strength between neurons
        a : float
            Threshold parameter
        B : numpy.ndarray
            Rotation matrix for coupling
        G : numpy.ndarray
            Adjacency matrix for network connectivity
        use_adaptive_params : bool
            Whether to use adaptive parameters for heterogeneous neuron populations
        noise_level : float
            Level of noise to add to the simulation (0.0-1.0)
        """
        self.N = N  # Number of neurons
        
        # Handle adaptive parameters for heterogeneous neuron populations
        if use_adaptive_params:
            # Create parameter distributions for different neuron types
            self.epsilon = np.random.normal(epsilon, epsilon * 0.1, N)  # Varied time scales
            self.sigma = np.random.normal(sigma, sigma * 0.05, N)  # Varied coupling strengths
            self.a = np.random.normal(a, 0.05, N)  # Varied threshold parameters
        else:
            # Use uniform parameters
            self.epsilon = np.ones(N) * epsilon
            self.sigma = np.ones(N) * sigma
            self.a = np.ones(N) * a
        
        self.B = B  # Rotation matrix
        self.G = G  # Adjacency matrix
        self.noise_level = noise_level  # Noise level for simulation
        
        logger.log("Instantiated enhanced Fitzhugh-Nagumo model", LogLevel.INFO)

    def _model(self, t, y):
        """
        Differential equations for the Fitzhugh-Nagumo model.
        
        Parameters:
        -----------
        t : float
            Current time point
        y : numpy.ndarray
            Current state vector [u1, u2, ..., uN, v1, v2, ..., vN]
            
        Returns:
        --------
        numpy.ndarray
            Rate of change for each state variable
        """
        N = len(y) // 2
        u = y[:N]  # Excitatory variable
        v = y[N:]  # Recovery variable
        
        du = np.zeros(N)
        dv = np.zeros(N)
        
        # Add noise if specified
        noise_u = np.zeros(N)
        noise_v = np.zeros(N)
        if self.noise_level > 0:
            noise_u = np.random.normal(0, self.noise_level, N)
            noise_v = np.random.normal(0, self.noise_level * 0.5, N)  # Less noise for recovery variable
        
        # Compute dynamics with coupling and noise
        for k in range(N):
            # Calculate coupling terms
            coupling_u = self.sigma[k] if isinstance(self.sigma, np.ndarray) else self.sigma
            coupling_u *= np.sum(self.G[k, :] * (self.B[0, 0] * u + self.B[0, 1] * v))
            
            coupling_v = self.sigma[k] if isinstance(self.sigma, np.ndarray) else self.sigma
            coupling_v *= np.sum(self.G[k, :] * (self.B[1, 0] * u + self.B[1, 1] * v))
            
            # Fitzhugh-Nagumo equations with coupling and noise
            du[k] = (u[k] - u[k]**3 / 3 - v[k] + coupling_u + noise_u[k]) / self.epsilon[k] if isinstance(self.epsilon, np.ndarray) else self.epsilon
            dv[k] = u[k] + self.a[k] + coupling_v + noise_v[k] if isinstance(self.a, np.ndarray) else self.a
        
        return np.concatenate([du, dv])

    def _generate_synthetic_data(self, T=200, dt=0.1, use_saved_models=True):
        """
        Generate synthetic data using the Fitzhugh-Nagumo model.
        
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
            t_points, v_points = self._load_datas(Constants.FN)
        else:
            # Initial conditions with controlled randomization
            u0 = np.random.uniform(-0.5, 0.5, self.N)  # Initial excitatory variables
            v0 = np.random.uniform(-0.5, 0.5, self.N)  # Initial recovery variables
            y0 = np.concatenate([u0, v0])  # Combined initial state

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
            
            logger.log("Generated synthetic data for enhanced Fitzhugh-Nagumo model", LogLevel.INFO)
            self._save_datas(Constants.FN, t_points, v_points)

        return t_points, v_points


class FitzhughNagumoNetworkModel(nn.Module):
    """
    Neural network model for forecasting Fitzhugh-Nagumo neuron dynamics using
    a combination of recurrent and feed-forward layers.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2, use_batch_norm=True):
        """
        Initialize the Fitzhugh-Nagumo network model.
        
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
        super(FitzhughNagumoNetworkModel, self).__init__()
        
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