import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataHandler:

    @staticmethod
    def generate_adjacency_matrix(base_matrix, n):
        G = base_matrix.copy()
        for _ in range(n - 1):
            G = np.kron(G, base_matrix)
        np.fill_diagonal(G, 0)
        return G / np.mean(G[G > 0])

    @staticmethod
    def _prepare_data(data, seq_length=50, corrections = False):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length])
            y.append(scaled_data[i+seq_length])
    
        X = np.array(X)
        y = np.array(y)
        
        return X,y
        
    @staticmethod
    def _split_train_test( X, y, train_ratio=0.9):
      
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