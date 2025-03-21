import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from neuronModels.FitzhughNagumoModel import FitzhughNagumoModel
from neuronModels.GRUNetwork import GRUNetwork
from neuronModels.HybridModel import HybridModel
from neuronModels.IzhikevichModel import izhikevich_model
from neuronModels.LifModel import LifModel
from utils.DataHandler import DataHandler
from utils.Logger import Logger, LogLevel
from utils.NoiseGenerator import NoiseGenerator
from utils.Plotter import Plotter
from utils.Constants import Constants

logger = Logger()

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)



def main():

 ################################# CREATING FN MODEL #################################

    logger.log("Generating synthetic data from FitzHugh-Nagumo model", LogLevel.INFO)
    
    fn_model = FitzhughNagumoModel()
    t_points, clean_data = fn_model._generate_synthetic_data(T=200, dt=0.1, use_saved_models=True)

### PLOTTING DYNAMICS AND CHIMERAS ###

    Plotter.plot_data(clean_data, index = 0, model_name= Constants.FN)
    Plotter.plot_phase_space(clean_data, var_indices=(0,20),time_steps=1000)
    Plotter.plot_results_chimera_analysis(t_points=t_points,v_points=clean_data)
    Plotter.plot_synchronization_map(clean_data[:,20:],t_points=t_points)

###GENERATING NOISY DATAS###

    noisy_data = clean_data.copy()
    noisy_data = DataHandler._generate_real_data(noisy_data)
    logger.log("Preparing data for the hybrid model", LogLevel.INFO)
    X_train, y_train, X_test, y_test, scaler = DataHandler.prepare_data_for_hybrid_model(clean_data, noisy_data)
    
###CREATING AND TRAINING HYBRIDMODEL###

    logger.log("Creating and training the hybrid model", LogLevel.INFO)
    hybrid_model = HybridModel()
    input_size = X_train.shape[2]  # Number of features
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
    # Create GRU network
    gru_model = hybrid_model._create_gru(input_size, output_size, hidden_size=64)
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
    # Define assimilation window (first 20% of the sequence)
    assimilation_window = int(0.2 * len(X_test))

    # Train the model
    trained_model = hybrid_model._train(
        gru_model, 
        X_train, 
        y_train, 
        optimizer, 
        epochs=100, 
        model_name= Constants.FN + "_train", 
        use_saved_model=True,
        assimilation_window=assimilation_window
    )
    
###FORECASTING###

    logger.log("Forecasting using the hybrid model", LogLevel.INFO)
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
    hybrid_predictions = scaler.inverse_transform(hybrid_predictions)

    #######STATISTICS ########


    Plotter.plot_forecast(t_points=t_points, noisy_data=noisy_data,physical_predictions=physical_predictions, hybrid_predictions=hybrid_predictions,assimilation_window=assimilation_window, model_name=Constants.FN
    )
    DataHandler.calculate_statistichs(physical_predictions=physical_predictions,noisy_data=noisy_data,hybrid_predictions=hybrid_predictions,assimilation_window=assimilation_window)
    
    
    ################################# CREATING LIF MODEL #################################


    logger.log("Generating synthetic data from Lif Model", LogLevel.INFO)
    lif_model = LifModel()
    t_points, clean_data = lif_model._generate_synthetic_data(use_saved_model=False)
    Plotter.plot_data(clean_data, index=0)

 ###GENERATING NOISY DATAS###

    noisy_data = clean_data.copy()
    noisy_data = DataHandler._generate_real_data(noisy_data)
    
    
    logger.log("Preparing data for the hybrid model", LogLevel.INFO)
    X_train, y_train, X_test, y_test, scaler = DataHandler.prepare_data_for_hybrid_model(clean_data, noisy_data)
    
   ###CREATING AND TRAINING HYBRIDMODEL###
    logger.log("Creating and training the hybrid model", LogLevel.INFO)
    hybrid_model = HybridModel()
    
    input_size = X_train.shape[2]  # Number of features
    output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    # Create GRU network
    gru_model = hybrid_model._create_gru(input_size, output_size, hidden_size=64)
    
    
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)
    
    # Define assimilation window (first 20% of the sequence)
    assimilation_window = int(0.2 * len(X_test))
    
    # Train the model
    trained_model = hybrid_model._train(
        gru_model, 
        X_train, 
        y_train, 
        optimizer, 
        epochs=100, 
        model_name=Constants.LIF + '_train', 
        use_saved_model=True,
        assimilation_window=assimilation_window
    )
    
###FORECASTING###
    logger.log("Forecasting using the hybrid model", LogLevel.INFO)
    

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
    
    
    hybrid_predictions = scaler.inverse_transform(hybrid_predictions)
    
###STATISTICS###
    Plotter.plot_forecast(t_points=t_points, noisy_data=noisy_data,physical_predictions=physical_predictions, hybrid_predictions=hybrid_predictions,assimilation_window=assimilation_window, model_name=Constants.LIF
    )
      
    DataHandler.calculate_statistichs(physical_predictions=physical_predictions,noisy_data=noisy_data,hybrid_predictions=hybrid_predictions,assimilation_window=assimilation_window)
    


    
if __name__ == "__main__":
    
    main()