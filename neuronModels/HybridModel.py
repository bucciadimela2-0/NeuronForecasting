
import os

import numpy as np
import torch
import torch.nn as nn

from neuronModels.GRUNetwork import GRUNetwork
from utils.Logger import Logger, LogLevel

logger = Logger()
class HybridModel:

    

    def _create_gru(self,input_size, output_size, hidden_size = 64, same_size = True, num_layers = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if not same_size else input_size

        # Create GRU network with correct input and output sizes
        model = GRUNetwork(input_size, hidden_size, self.output_size)

        logger.log("Created GRU network", LogLevel.INFO)
        
        return model
   

    def _train(self, model, x_train, y_train, optimizer, epochs, model_name=None, use_saved_model=True, assimilation_window=None):
       
        #TODO controlla se la cartella dove salverai il modello esiste, nel caso creata 
        #logger.log(f"Creata la directory per salvare i modelli: {self.MODEL_PATH}")

        #TODO se flag a true ed esiste il modello nella cartella caricalo
        #logger.log(f"Caricando modello salvato da {model_path}...")

        logger.log("Training started..", LogLevel.INFO)
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            optimizer.zero_grad()
            # Forward pass on training data
            outputs = model(x_train)

            loss = nn.MSELoss()(outputs, y_train)
            # Backpropagation
            loss.backward()
            optimizer.step()

            # Validation on test set: si sta facendo double dipping, meglio spezzare in valmode
            if (epoch + 1) % 20 == 0:
                logger.log(f'Epoch [{epoch+1}/{epochs}], '
                    f'Train Loss: {loss.item():.4f} ')
        
        logger.log("Training ended", LogLevel.INFO)
        #TODO salva il modello
        
       # logger.log(f"Modello salvato in {model_path}")

        

        return model

    def _forecast(self, model, x_test, y_test, physical_predictions=None, assimilation_window=None, autonomous_forecast=True):
        """
        Forecast using trained hybrid model with data assimilation and autonomous prediction.
        
        Parameters:
        -----------
        model : GRUNetwork
            The trained neural network model
        x_test : torch.Tensor
            Input sequences from physical model predictions
        y_test : torch.Tensor
            Target error corrections or true values
        physical_predictions : numpy.ndarray, optional
            Raw predictions from the physical model
        assimilation_window : int, optional
            Number of time steps for data assimilation during forecasting
        autonomous_forecast : bool
            Whether to run in autonomous mode after assimilation window
            
        Returns:
        --------
        numpy.ndarray
            Corrected predictions from the hybrid model
        """
        # Convert tensors to numpy for processing
        if isinstance(x_test, torch.Tensor):
            x_test_np = x_test.numpy()
        else:
            x_test_np = x_test.copy()
            
        if isinstance(y_test, torch.Tensor):
            y_test_np = y_test.numpy()
        else:
            y_test_np = y_test.copy()
            
        # If physical predictions not provided, use x_test as physical predictions
        if physical_predictions is None:
            physical_predictions = x_test_np
            
        # Initialize the corrected predictions array
        corrected_predictions = np.zeros_like(physical_predictions)
        
        # Apply data assimilation for the initial window if specified
        if assimilation_window is not None and assimilation_window > 0:
            corrected_predictions[:assimilation_window] = y_test_np[:assimilation_window]
            start_idx = assimilation_window
        else:
            start_idx = 0
            
        # For autonomous forecasting, we need to iteratively apply corrections
        if autonomous_forecast and start_idx < len(physical_predictions):
            # Initialize the current sequence with the last window of assimilated states
            seq_length = x_test.shape[1]  # Get original sequence length
            current_sequence = x_test_np[start_idx :start_idx + seq_length ]
            
            # Iteratively predict and update for each time step
            for t in range(start_idx, len(physical_predictions)):
                # Convert current sequence to tensor with proper dimensions (batch, sequence, features)
                current_input = torch.tensor(current_sequence, dtype=torch.float32)
                
                # Predict correction using the neural network
                with torch.no_grad():
                    print(current_input.shape)
                    correction = model(current_input).numpy()  # Get first batch prediction
                
                # Get physical model prediction and apply correction
                phys_pred = physical_predictions[t:t+1]
                corrected = phys_pred + correction
                corrected_predictions[t] = corrected[0]  # Store the corrected prediction
                
                
                
                if len(current_sequence.shape) == 3:
   

                    # Update sequence for next iteration by removing oldest and adding newest prediction
                    current_sequence = np.vstack((current_sequence[1:, -1, :], phys_pred.reshape(1, -1)))
                # Reshape to maintain 3D structure (batch, sequence, features)
                    current_sequence = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        else:
            # For non-autonomous mode, simply apply corrections in one batch
            with torch.no_grad():
                corrections = model(torch.tensor(x_test_np, dtype=torch.float32)).numpy()
                corrected_predictions[start_idx:] = physical_predictions[start_idx:] + corrections[start_idx:]
        
        logger.log("Hybrid model forecast completed", LogLevel.INFO)
        return corrected_predictions

    def _standardize(self, data):
        data = self.scaler.transform(data)
        return data
    
            

            