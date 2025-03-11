
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from NeuronModel.GRUNetwork import GRUNetwork


class HybridModel_FN:
    def __init__(self,N,epsilon,sigma, a, B, G):
        self.N = N
        self.sigma = sigma
        self.epsilon = epsilon
        self.a = a
        self.B = B
        self.G = G
        self.scaler = StandardScaler()
        self.scaler2 = MinMaxScaler()
        
        
    
    def _model_fitzhughNagumo(self,t,y):
        
        N = len(y) // 2
        u = y[:N]
        v = y[N:]
        du = np.zeros(N)
        dv = np.zeros(N)
        for k in range(N):
            coupling_u = self.sigma * np.sum(self.G[k, :] * (self.B[0, 0] * u + self.B[0, 1] * v))
            coupling_v = self.sigma * np.sum(self.G[k, :] * (self.B[1, 0] * u + self.B[1, 1] * v))
            du[k] = (u[k] - u[k]**3 / 3 - v[k] + coupling_u) / self.epsilon
            dv[k] = u[k] + self.a + coupling_v
        return np.concatenate([du, dv])

    #Forse non necessario
    def _create_gru(self,input_size, output_size, hidden_size = 64, same_size = True, num_layers = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        if same_size:
            self.output_size = input_size

        model = GRUNetwork(input_size, hidden_size, output_size)
        
        return model

    def _generate_synthetic_data(self, T=100, dt=0.1):
        """
        Generate training data using physical model
        """
        # Initial conditions
        #mhm...
        y0 = np.random.uniform(-1, 1, self.N * 2)


        # Time evaluation points
        t_eval = np.arange(0, T, dt)

        # Solve the initial value problem
        sol = solve_ivp(
            self._model_fitzhughNagumo,
            [0,T],
            y0,
            method='RK45',
            t_eval=t_eval, vectorized= True
        )

        return sol.t, sol.y.T

    def _prepare_data(self, data, seq_length=20, corrections = False):
        
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        
       
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
    
        X = np.array(X)
        y = np.array(y)
        
        return X,y

    def _split_train_test(self, X, y, train_ratio=0.8):
      
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

    def create_error_sequences(self,model_data, true_data, seq_length=20):
        model_data = self.scaler.transform(model_data)
        true_data = self.scaler.transform(true_data)
        error_sequences, error_targets = [], []
        errors = true_data - model_data
        for i in range(len(errors) - seq_length):
            error_sequences.append(errors[i:i+seq_length])
            error_targets.append(errors[i+seq_length])
        return torch.tensor(np.array(error_sequences), dtype=torch.float32), torch.tensor(np.array(error_targets), dtype=torch.float32)

   

    def _physical_loss(self, predicted, actual,restriction = False):
        """
        Calculate physics-informed loss
        Combines ML prediction error with ODE model constraints
        """
        ml_loss = nn.MSELoss()(predicted, actual)
        #ml_loss = nn.DWTloss()(predicted,actual)

        if restriction:
            physics_penalty = self._calculate_penality(predicted)
            ml_loss = ml_loss  + physics_penalty

        return ml_loss

    def _calculate_penality(self,predicted):
        # controllare cosa fa
        t_eval = np.linspace(0, 1, predicted.shape[0])  # Create evenly spaced time points
        sol = solve_ivp(
            self._model_fitzhughNagumo,  # ODE function
            [0, 1],  # Time span from 0 to 1
            predicted[0].detach().numpy(),  # Initial condition (first time step of prediction)
            t_eval=t_eval,  # Specific time points to evaluate
             method='RK45',
            vectorized = True
        )
        penalty = np.mean(np.abs(sol.y.T - predicted.detach().numpy()))
        return penalty

    #ricorda di creare e passare modello + ottimizzatore
    def _train(self, model, x_train, y_train, optimizer,epochs = 100, restriction = False ):
        '''
        X_train, y_train, X_test, y_test, scaler = self._prepare_data(
        data,
        train_ratio=train_ratio)
        '''

        for epoch in range(epochs):
            model.train()  # Set model to training mode
            optimizer.zero_grad()
            # Forward pass on training data
            outputs = model(x_train)

            # Hybrid loss calculation
            loss = self._physical_loss(outputs, y_train, restriction=restriction)

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

        #forecasted = self.scaler.transform(forecasted)
      
        # Inverse transform
        return forecasted

    def _standardize(self, data):
        data = self.scaler.transform(data)
        return data
    
            

            