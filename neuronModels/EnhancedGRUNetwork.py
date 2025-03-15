import torch
import torch.nn as nn


class EnhancedGRUNetwork(nn.Module):
    """
    Enhanced GRU network with multiple layers, dropout, and batch normalization
    for improved performance in neuronal time series forecasting.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2, use_batch_norm=True):
        """
        Initialize the enhanced GRU network.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of hidden units in each GRU layer
        output_size : int
            Number of output features
        num_layers : int
            Number of GRU layers
        dropout : float
            Dropout probability (0-1)
        use_batch_norm : bool
            Whether to use batch normalization
        """
        super(EnhancedGRUNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        
        # GRU layers
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