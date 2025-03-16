import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from neuronModels.FitzhughNagumoModel import FitzhughNagumoModel
from neuronModels.IzhikevichModel import izhikevich_model
from neuronModels.LifModel import LifModel
from neuronModels.HybridModel import HybridModel
from neuronModels.GRUNetwork import GRUNetwork
from utils.NoiseGenerator import NoiseGenerator
from utils.DataHandler import DataHandler
from utils.Logger import Logger, LogLevel

logger = Logger()

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)



def main():
    # 1. Generate synthetic data from a physical model (FitzHugh-Nagumo)
    

    logger.log("Generating synthetic data from FitzHugh-Nagumo model", LogLevel.INFO)
    
    fn_model = FitzhughNagumoModel()
    t_points, clean_data = fn_model._generate_synthetic_data(T=200, dt=0.1, use_saved_models=True)
    
    # 2. Add biological noise to create simulated real-world data
    logger.log("Adding biological noise to create simulated real-world data", LogLevel.INFO)
    noisy_data = clean_data.copy()
    noisy_data = DataHandler._generate_real_data(noisy_data)
    
    # 3. Prepare data for the hybrid model
    logger.log("Preparing data for the hybrid model", LogLevel.INFO)
    X_train, y_train, X_test, y_test, scaler = DataHandler.prepare_data_for_hybrid_model(clean_data, noisy_data)
    
    # 4. Create and train the hybrid model
    logger.log("Creating and training the hybrid model", LogLevel.INFO)
    hybrid_model = HybridModel()
    
    # Get input and output dimensions
    input_size = X_train.shape[2]  # Number of features
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    # Create GRU network
    gru_model = hybrid_model._create_gru(input_size, output_size, hidden_size=64)
    
    # Define optimizer
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
    
    # Define assimilation window (e.g., first 20% of the sequence)
    assimilation_window = int(0.2 * len(X_test))
    
    # Train the model
    trained_model = hybrid_model._train(
        gru_model, 
        X_train, 
        y_train, 
        optimizer, 
        epochs=500, 
        model_name="HybridFN", 
        use_saved_model=True,
        assimilation_window=assimilation_window
    )
    
    # 5. Forecast using the hybrid model
    logger.log("Forecasting using the hybrid model", LogLevel.INFO)
    
    # Convert test data back to numpy for physical model
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()
    
    physical_predictions=X_test_np[:, -1, :]
    
    # Forecast with hybrid model
    hybrid_predictions = hybrid_model._forecast(
        trained_model,
        X_test,
        y_test,
        physical_predictions=X_test_np[:, -1, :],  # Use last time step of each sequence
        assimilation_window=assimilation_window,
        autonomous_forecast=True
    )
    
    # Convert predictions back to original scale
    hybrid_predictions = scaler.inverse_transform(hybrid_predictions)
    
    # 6. Evaluate and visualize results
    logger.log("Evaluating and visualizing results", LogLevel.INFO)
    
    # Calculate mean squared error
    physical_mse = np.mean((physical_predictions - noisy_data[-len(physical_predictions):]) ** 2)
    hybrid_mse = np.mean((hybrid_predictions - noisy_data[-len(hybrid_predictions):]) ** 2)
    
    logger.log(f"Physical model MSE: {physical_mse:.6f}", LogLevel.INFO)
    logger.log(f"Hybrid model MSE: {hybrid_mse:.6f}", LogLevel.INFO)
    logger.log(f"Improvement: {(1 - hybrid_mse/physical_mse) * 100:.2f}%", LogLevel.INFO)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Select a representative neuron to visualize (e.g., first neuron)
    neuron_idx = 0
    
    # Plot time series
    plt.subplot(2, 1, 1)
    plt.plot(t_points[-len(noisy_data):], noisy_data[:, neuron_idx], 'k-', label='True Values')
    plt.plot(t_points[-len(physical_predictions):], physical_predictions[:, neuron_idx], 'b--', label='Physical Model')
    plt.plot(t_points[-len(hybrid_predictions):], hybrid_predictions[:, neuron_idx], 'r-', label='Hybrid Model')
    
    # Add vertical line to mark end of assimilation window
    if assimilation_window > 0:
        assimilation_time = t_points[-len(hybrid_predictions) + assimilation_window]
        plt.axvline(x=assimilation_time, color='g', linestyle='--', label='End of Assimilation')
    
    plt.title('Comparison of Forecasting Methods')
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential')
    plt.legend()
    
    # Plot error
    plt.subplot(2, 1, 2)
    physical_error = np.abs(physical_predictions[:, neuron_idx] - noisy_data[-len(physical_predictions):, neuron_idx])
    hybrid_error = np.abs(hybrid_predictions[:, neuron_idx] - noisy_data[-len(hybrid_predictions):, neuron_idx])
    
    plt.plot(t_points[-len(physical_error):], physical_error, 'b--', label='Physical Model Error')
    plt.plot(t_points[-len(hybrid_error):], hybrid_error, 'r-', label='Hybrid Model Error')
    
    # Add vertical line to mark end of assimilation window
    if assimilation_window > 0:
        plt.axvline(x=assimilation_time, color='g', linestyle='--', label='End of Assimilation')
    
    plt.title('Absolute Error')
    plt.xlabel('Time')
    plt.ylabel('Error Magnitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('figures/hybrid_model_forecast.png')
    plt.show()
    
if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    import os
    os.makedirs("figures", exist_ok=True)
    
    main()