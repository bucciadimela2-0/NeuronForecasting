
import os

import numpy as np
import torch
import torch.nn as nn

from neuronModels.GRUNetwork import GRUNetwork
from utils.Logger import Logger, LogLevel

logger = Logger()
class HybridModel:

    MODEL_PATH="saved_models"

    def _create_gru(self,input_size, output_size, hidden_size = 64, same_size = True, num_layers = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        if same_size:
            self.output_size = input_size

        model = GRUNetwork(input_size, hidden_size, output_size)
        
        return model
   

    def _train(self, model, x_train, y_train, optimizer,epochs, model_name, use_saved_model=True):
        '''
        X_train, y_train, X_test, y_test, scaler = self._prepare_data(
        data,
        train_ratio=train_ratio)
        '''
        #TODO controlla se la cartella dove salverai il modello esiste, nel caso creata 
        logger.log(f"Creata la directory per salvare i modelli: {self.MODEL_PATH}")

        #TODO se flag a true ed esiste il modello nella cartella caricalo
        logger.log(f"Caricando modello salvato da {model_path}...")

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
        
        #TODO salva il modello
        
        logger.log(f"Modello salvato in {model_path}")

        return model

    def _forecast(self, model, x_test ,y_test, corrections = False):
        """
        Forecast using trained hybrid model
        """

        # Forecast
        with torch.no_grad():
            forecasted = model(x_test).numpy()
            y_test = y_test.numpy()

        if corrections:
            corrected_predictions = (y_test + forecasted) 
            #corrected_predictions = self.scaler.transform(corrected_predictions)
            return corrected_predictions

        
      
        # Inverse transform
        return forecasted

    def _standardize(self, data):
        data = self.scaler.transform(data)
        return data
    
            

            