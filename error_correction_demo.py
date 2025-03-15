import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neuronModels.FitzhughNagumoModel import FitzhughNagumoModel
from neuronModels.GRUNetwork import GRUNetwork
from utils.NoiseGenerator import NoiseGenerator
from utils.DataHandler import DataHandler
from utils.Plotter import Plotter
from utils.Constants import Constants
from utils.Logger import Logger, LogLevel

logger = Logger()


def main():
    # Step 1: Generate synthetic data from the FitzHugh-Nagumo model
    logger.log("Starting error correction demonstration", LogLevel.INFO)
    fn_model = FitzhughNagumoModel()
    t_points, clean_data = fn_model._generate_synthetic_data(T=100, dt=0.1, use_saved_models=True)
    
    # Step 2: Add realistic measurement noise to simulate real-world data
    logger.log("Adding measurement noise to synthetic data", LogLevel.INFO)
    noisy_data = NoiseGenerator.add_measurement_noise(clean_data, noise_level=0.05, noise_type='gaussian')
    
    # Also add some systematic error to make it more realistic
    noisy_data = NoiseGenerator.add_systematic_error(noisy_data, bias=0.02, drift_factor=0.001)
    
    # Step 3: Prepare data for training the error correction model
    # We'll use the clean data as ground truth and noisy data as input
    logger.log("Preparing data for error correction model", LogLevel.INFO)
    
    # Create error sequences (difference between true and noisy data)
    error_X, error_y = DataHandler.create_error_sequences(noisy_data, clean_data, seq_length=50)
    
    # Split into training and testing sets
    split_idx = int(len(error_X) * 0.8)
    X_train, y_train = error_X[:split_idx], error_y[:split_idx]
    X_test, y_test = error_X[split_idx:], error_y[split_idx:]
    
    # Step 4: Create and train the GRU model for error correction
    logger.log("Training GRU model for error correction", LogLevel.INFO)
    input_size = X_train.shape[2]  # Number of features (neurons)
    hidden_size = 64
    output_size = y_train.shape[1]  # Number of output features
    
    error_model = GRUNetwork(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(error_model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 100
    for epoch in range(epochs):
        error_model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = error_model(X_train)
        loss = nn.MSELoss()(outputs, y_train)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            logger.log(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}', LogLevel.INFO)
    
    # Step 5: Evaluate the error correction model
    logger.log("Evaluating error correction model", LogLevel.INFO)
    error_model.eval()
    with torch.no_grad():
        predicted_errors = error_model(X_test).numpy()
        actual_errors = y_test.numpy()
    
    # Step 6: Apply error correction to the noisy test data
    test_idx = split_idx + 50  # Account for sequence length
    noisy_test_data = noisy_data[test_idx:test_idx+len(predicted_errors)]
    corrected_data = noisy_test_data + predicted_errors
    clean_test_data = clean_data[test_idx:test_idx+len(predicted_errors)]
    
    # Step 7: Visualize the results
    logger.log("Visualizing results", LogLevel.INFO)
    
    # Plot a sample neuron
    neuron_idx = 0  # Choose which neuron to visualize
    time_steps = 200  # Number of time steps to show
    
    plt.figure(figsize=(12, 6))
    plt.plot(clean_test_data[:time_steps, neuron_idx], label='True Data', color='green')
    plt.plot(noisy_test_data[:time_steps, neuron_idx], label='Noisy Data', color='red', alpha=0.6)
    plt.plot(corrected_data[:time_steps, neuron_idx], label='Corrected Data', color='blue', linestyle='--')
    plt.title(f'Error Correction for Neuron {neuron_idx}')
    plt.xlabel('Time Steps')
    plt.ylabel('Membrane Potential')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print error metrics
    noisy_mse = np.mean((noisy_test_data - clean_test_data) ** 2)
    corrected_mse = np.mean((corrected_data - clean_test_data) ** 2)
    improvement = (1 - corrected_mse / noisy_mse) * 100
    
    logger.log(f"Noisy data MSE: {noisy_mse:.6f}", LogLevel.INFO)
    logger.log(f"Corrected data MSE: {corrected_mse:.6f}", LogLevel.INFO)
    logger.log(f"Error reduction: {improvement:.2f}%", LogLevel.INFO)
    
    # Also use the Plotter class to visualize multiple neurons
    Plotter.plot_data(data=clean_test_data, 
                     forecasted=corrected_data, 
                     num_series=3,  # Show 3 neurons
                     time_steps=200, 
                     var_name='Neuron ', 
                     title='Error Correction Results')


if __name__ == "__main__":
    main()