
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from neuronModels.GRUNetwork import GRUNetwork
from utils.Logger import Logger, LogLevel

logger = Logger()
class HybridModel:

    MODEL_PATH="saved_models"

    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler2 = MinMaxScaler()
        

    def _create_gru(self,input_size, output_size, hidden_size = 64, same_size = True, num_layers = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        if same_size:
            self.output_size = input_size

        model = GRUNetwork(input_size, hidden_size, output_size)
        
        return model
   

    def _prepare_data(self, data, seq_length=50, corrections = False):
        #FIXME :usare scaled_data e vedere che i modelli funzionano
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        
       
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(scaled_data[i+seq_length])
    
        X = np.array(X)
        y = np.array(y)
        
        return X,y
        

    def _split_train_test(self, X, y, train_ratio=0.9):
      
        split_idx = int(len(X) * train_ratio)
        
        
        
        # Split data
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
     
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train,dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test,dtype=torch.float32)
        
        return X_train, y_train, X_test, y_test

    def create_error_sequences(self,model_data, true_data, seq_length=50):
        model_data = self.scaler.transform(model_data)
        true_data = self.scaler.transform(true_data)
        error_sequences, error_targets = [], []
        errors = true_data - model_data
        for i in range(len(errors) - seq_length):
            error_sequences.append(errors[i:i+seq_length])
            error_targets.append(errors[i+seq_length])
        return torch.tensor(np.array(error_sequences), dtype=torch.float32), torch.tensor(np.array(error_targets), dtype=torch.float32)

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

        #forecasted = self.scaler.transform(forecasted)
      
        # Inverse transform
        return forecasted

    def _standardize(self, data):
        data = self.scaler.transform(data)
        return data
    
            

            