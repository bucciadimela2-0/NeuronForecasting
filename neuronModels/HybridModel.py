
import numpy as np
import torch
import torch.nn as nn

from neuronModels.GRUNetwork import GRUNetwork


class HybridModel:

    def _create_gru(self,input_size, output_size, hidden_size = 64, same_size = True, num_layers = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        if same_size:
            self.output_size = input_size

        model = GRUNetwork(input_size, hidden_size, output_size)
        
        return model
   

    def _train(self, model, x_train, y_train, optimizer,epochs = 100, restriction = False ):
       

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
               
                print(f'Epoch [{epoch+1}/{epochs}], '
                    f'Train Loss: {loss.item():.4f} ')

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
    
            

            