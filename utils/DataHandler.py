import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.Logger import Logger, LogLevel
from utils.NoiseGenerator import NoiseGenerator

logger = Logger()


class DataHandler:

    @staticmethod
    def generate_adjacency_matrix(base_matrix, n):
        G = base_matrix.copy()
        for _ in range(n - 1):
            G = np.kron(G, base_matrix)
        np.fill_diagonal(G, 0)
        return G / np.mean(G[G > 0])

    @staticmethod
    def create_error_sequences(model_data, true_data, seq_length=50):
        scaler = StandardScaler()
        model_data = scaler.fit_transform(model_data)
        true_data = scaler.transform(true_data)
        error_sequences, error_targets = [], []
        errors = true_data - model_data
        for i in range(len(errors) - seq_length):
            error_sequences.append(errors[i:i+seq_length])
            error_targets.append(errors[i+seq_length])
        return torch.tensor(np.array(error_sequences), dtype=torch.float32), torch.tensor(np.array(error_targets), dtype=torch.float32)
    
    @staticmethod
    def _generate_real_data(clean_data):

        logger.log("Adding biological noise to create simulated real-world data")
        noisy_data = NoiseGenerator.add_measurement_noise(clean_data, noise_level=0.05, noise_type='gaussian')
        
        # Add more realistic biological noise
        noisy_data = NoiseGenerator.add_pink_noise(noisy_data, noise_level=0.03)
        noisy_data = NoiseGenerator.add_synaptic_noise(noisy_data, noise_level=0.02, tau=15)
        noisy_data = NoiseGenerator.add_periodic_artifact(noisy_data, amplitude=0.02, frequency=0.05)
        
        # Also add some systematic error to make it more realistic
        noisy_data = NoiseGenerator.add_systematic_error(noisy_data, bias=0.02, drift_factor=0.001)
        return noisy_data

    @staticmethod
    def prepare_data_for_hybrid_model(clean_data, noisy_data, seq_length=50):
       
        # Standardize data
        scaler = StandardScaler()
        clean_scaled = scaler.fit_transform(clean_data)
        noisy_scaled = scaler.transform(noisy_data)
        
        # Calculate error between physical model and true data
        error_data = noisy_scaled - clean_scaled
        
        # Create sequences for training
        X, y = [], []
        for i in range(len(clean_scaled) - seq_length):
            X.append(clean_scaled[i:i+seq_length])
            y.append(error_data[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        train_ratio = 0.8
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        return X_train, y_train, X_test, y_test, scaler

    @staticmethod
    def save_plot(fig, filename, output_dir='plots', dpi=300, format='png', 
              transparent=False, bbox_inches='tight', pad_inches=0.1):
       
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.log(f"Created directory: {output_dir}")
        
        
        full_path = os.path.join(output_dir, f"{filename}.{format}")
        
       
        fig.savefig(full_path, dpi=dpi, format=format, transparent=transparent,
                    bbox_inches=bbox_inches, pad_inches=pad_inches)
        
        logger.log(f"Figure saved to: {full_path}")
        return full_path
        
    def calculate_statistichs(physical_predictions, noisy_data, hybrid_predictions, assimilation_window):
        
        logger.log("Evaluating and visualizing results", LogLevel.INFO)
        
        physical_mse = np.mean((physical_predictions[-assimilation_window:] - noisy_data[-assimilation_window:]) ** 2)
        hybrid_mse = np.mean((hybrid_predictions[-assimilation_window:] - noisy_data[-assimilation_window:]) ** 2)
        
        logger.log(f"Physical model MSE: {physical_mse:.6f}", LogLevel.INFO)
        logger.log(f"Hybrid model MSE: {hybrid_mse:.6f}", LogLevel.INFO)
        logger.log(f"Improvement: {(1 - hybrid_mse/physical_mse) * 100:.2f}%", LogLevel.INFO)

    def save_model(model, model_name, model_dir='neuronModels/SavedModels', include_timestamp=False):
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.log(f"Created directory: {model_dir}", LogLevel.INFO)
        
        # Generate filename with optional timestamp
       
        filename = f"{model_name}.pt"
        
        # Construct full path
        full_path = os.path.join(model_dir, filename)
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            
            'hyperparameters': {
                'input_size': model.input_size if hasattr(model, 'input_size') else None,
                'hidden_size': model.hidden_size if hasattr(model, 'hidden_size') else None,
                'output_size': model.output_size if hasattr(model, 'output_size') else None,
                
            },
            
        }, full_path)
        
        logger.log(f"Model saved to: {full_path}", LogLevel.INFO)
        return full_path

    @staticmethod
    def load_model(model_class, model_path):
        try:
            # Ensure .pt extension
            checkpoint = torch.load(model_path + ".pt" if not model_path.endswith('.pt') else model_path)
            
            # Get hyperparameters
            hyperparams = checkpoint.get('hyperparameters', {})
            
            # Create model instance
            model = model_class(
                input_size=hyperparams.get('input_size'),
                hidden_size=hyperparams.get('hidden_size'),
                output_size=hyperparams.get('output_size')
            )
            
            # Load the state dictionary
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.log(f"Model loaded from: {model_path}", LogLevel.INFO)
            return model
        except Exception as e:
            logger.log(f"Error loading model: {e}", LogLevel.ERROR)
            raise
        