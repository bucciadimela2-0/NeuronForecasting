import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


class Plotter:
    @staticmethod
    def plot_data(data=None, forecasted=None, index=0, num_series=5, time_steps=200, 
                  var_name='u', title=None, figsize=(10, 5), y_test=None):
        """
        Visualize original and forecasted time series data for neuronal activity.
        
        This plot shows the comparison between original neuronal recordings and model predictions,
        allowing visual assessment of how well the model captures the dynamics of neuronal activity.
        It's useful for qualitative evaluation of model performance and identifying patterns or
        discrepancies in the forecasts.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Original time series data, shape (time_steps, n_neurons)
        forecasted : numpy.ndarray
            Forecasted time series data, shape (time_steps, n_neurons)
        index : int
            Index of the neuron to plot when showing a single series
        num_series : int
            Number of series to plot when showing multiple neurons
        time_steps : int
            Number of time steps to display
        var_name : str
            Variable name for the legend (e.g., 'u' for membrane potential)
        title : str
            Plot title
        figsize : tuple
            Figure size (width, height)
        y_test : numpy.ndarray
            Alternative to 'data' for test data comparison
        """
        plt.figure(figsize=figsize)
        
        # Handle both y_test/forecasted and data/forecasted cases
        if y_test is not None:
            # Single index comparison case
            plt.plot(y_test[:time_steps, index], 
                    label=f'Original {var_name}{index}')
            if forecasted is not None:
                plt.plot(forecasted[:time_steps, index], 
                        label=f'Forecasted {var_name}{index}',
                        color='red',
                        linestyle='dashed')
        else:
            # Multiple series case
            n_series = min(num_series, data.shape[1]) if len(data.shape) > 1 else 1
            if forecasted is not None:
                for i in range(n_series):
                    plt.plot(data[:time_steps, i], 
                            label=f'Original {var_name}{i}')
                    plt.plot(forecasted[:time_steps, i], 
                            label=f'Forecasted {var_name}{i}',
                            color='red',
                            linestyle='dashed')
            else:
                for i in range(n_series):
                    plt.plot(data[:time_steps, i], 
                            label=f'Original {var_name}{i}')

        plt.title(title or 'Comparison Plot')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_error_distribution(y_true, y_pred, figsize=(12, 5), title="Error Distribution Analysis"):
        """
        Visualize the distribution of prediction errors to assess model performance.
        
        This plot shows the statistical distribution of errors between true and predicted values,
        helping to identify if errors are normally distributed (good) or skewed/biased (problematic).
        It's useful for understanding the error characteristics and potential systematic biases in the model.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            Ground truth values
        y_pred : numpy.ndarray
            Predicted values
        figsize : tuple
            Figure size (width, height)
        title : str
            Plot title
        """
        errors = y_true.flatten() - y_pred.flatten()
        
        plt.figure(figsize=figsize)
        
        # Create a two-panel plot
        plt.subplot(1, 2, 1)
        sns.histplot(errors, kde=True)
        plt.title("Error Distribution")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.axvline(x=0, color='red', linestyle='--')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_true.flatten(), y=y_pred.flatten(), alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title("True vs Predicted Values")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_error_metrics(y_true, y_pred, neuron_indices=None, figsize=(14, 6)):
        """
        Visualize error metrics across different neurons to identify problematic predictions.
        
        This plot shows how prediction errors vary across different neurons, helping to identify
        which neurons are more difficult to predict accurately. It's useful for focusing improvement
        efforts on specific neurons or understanding if certain neuron types are systematically
        harder to model.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            Ground truth values, shape (time_steps, n_neurons)
        y_pred : numpy.ndarray
            Predicted values, shape (time_steps, n_neurons)
        neuron_indices : list or None
            Specific neuron indices to analyze, if None uses all neurons
        figsize : tuple
            Figure size (width, height)
        """
        if neuron_indices is None:
            neuron_indices = range(y_true.shape[1])
        
        mse_values = [mean_squared_error(y_true[:, i], y_pred[:, i]) for i in neuron_indices]
        mae_values = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in neuron_indices]
        
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(neuron_indices)), mse_values)
        plt.xticks(range(len(neuron_indices)), [f"Neuron {i}" for i in neuron_indices], rotation=45)
        plt.title("Mean Squared Error by Neuron")
        plt.ylabel("MSE")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(neuron_indices)), mae_values)
        plt.xticks(range(len(neuron_indices)), [f"Neuron {i}" for i in neuron_indices], rotation=45)
        plt.title("Mean Absolute Error by Neuron")
        plt.ylabel("MAE")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle("Error Metrics Across Neurons")
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_training_history(history, figsize=(12, 5)):
        """
        Visualize model training history to assess convergence and potential overfitting.
        
        This plot shows how the loss values change during training for both training and validation sets.
        It helps identify if the model is learning properly, converging, or potentially overfitting to the
        training data. A good training process shows decreasing losses that eventually stabilize without
        divergence between training and validation.
        
        Parameters:
        -----------
        history : dict or tensorflow.keras.callbacks.History
            Training history containing 'loss' and 'val_loss' keys
        figsize : tuple
            Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Handle both dictionary and keras History object
        if hasattr(history, 'history'):
            history = history.history
        
        # Plot training & validation loss values
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
            
        plt.title('Model Training History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_phase_portrait(data, forecasted=None, neuron_index=0, var1_idx=0, var2_idx=1, 
                           time_steps=200, figsize=(10, 8), title=None):
        """
        Visualize phase portraits of neuronal dynamics to analyze system behavior.
        
        This plot shows the relationship between two state variables (e.g., membrane potential and
        recovery variable) in a 2D phase space. It's particularly useful for analyzing the dynamical
        systems behavior of neurons, identifying attractors, limit cycles, or chaotic behavior, and
        comparing how well the model captures these dynamics.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Original time series data, shape (time_steps, n_neurons, n_variables)
        forecasted : numpy.ndarray or None
            Forecasted time series data, shape (time_steps, n_neurons, n_variables)
        neuron_index : int
            Index of the neuron to analyze
        var1_idx : int
            Index of the first state variable
        var2_idx : int
            Index of the second state variable
        time_steps : int
            Number of time steps to display
        figsize : tuple
            Figure size (width, height)
        title : str
            Plot title
        """
        plt.figure(figsize=figsize)
        
        # Plot original data phase portrait
        plt.plot(data[:time_steps, neuron_index, var1_idx], 
                data[:time_steps, neuron_index, var2_idx], 
                'b-', label='Original', alpha=0.7)
        plt.plot(data[0, neuron_index, var1_idx], 
                data[0, neuron_index, var2_idx], 
                'go', markersize=8, label='Start')
        
        # Plot forecasted data if provided
        if forecasted is not None:
            plt.plot(forecasted[:time_steps, neuron_index, var1_idx], 
                    forecasted[:time_steps, neuron_index, var2_idx], 
                    'r--', label='Forecasted', alpha=0.7)
        
        plt.title(title or f'Phase Portrait - Neuron {neuron_index}')
        plt.xlabel(f'Variable {var1_idx}')
        plt.ylabel(f'Variable {var2_idx}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_spectral_analysis(data, forecasted=None, neuron_index=0, time_steps=None, 
                             sampling_rate=1.0, figsize=(12, 6), title=None):
        """
        Perform spectral analysis to identify frequency components in neuronal signals.
        
        This plot shows the frequency domain representation of neuronal signals, helping to
        identify dominant frequencies, rhythms, and oscillatory patterns. It's useful for
        comparing the spectral characteristics of original and forecasted signals to assess
        how well the model captures the frequency components of neuronal activity.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Original time series data
        forecasted : numpy.ndarray or None
            Forecasted time series data
        neuron_index : int
            Index of the neuron to analyze
        time_steps : int or None
            Number of time steps to use, if None uses all available data
        sampling_rate : float
            Sampling rate of the data in Hz
        figsize : tuple
            Figure size (width, height)
        title : str
            Plot title
        """
        from scipy import signal
        
        plt.figure(figsize=figsize)
        
        # Use all time steps if not specified
        if time_steps is None:
            time_steps = data.shape[0]
        else:
            time_steps = min(time_steps, data.shape[0])
        
        # Extract the data for the specified neuron
        if data.ndim > 2:
            original_signal = data[:time_steps, neuron_index, 0]  # First variable
        else:
            original_signal = data[:time_steps, neuron_index]
        
        # Compute power spectral density of original signal
        f, Pxx = signal.welch(original_signal, fs=sampling_rate, nperseg=min(256, len(original_signal)))
        plt.semilogy(f, Pxx, 'b-', label='Original')
        
        # Compute power spectral density of forecasted signal if provided
        if forecasted is not None:
            if forecasted.ndim > 2:
                forecasted_signal = forecasted[:time_steps, neuron_index, 0]
            else:
                forecasted_signal = forecasted[:time_steps, neuron_index]
                
            f_pred, Pxx_pred = signal.welch(forecasted_signal, fs=sampling_rate, 
                                          nperseg=min(256, len(forecasted_signal)))
            plt.semilogy(f_pred, Pxx_pred, 'r--', label='Forecasted')
        
        plt.title(title or f'Spectral Analysis - Neuron {neuron_index}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_model_comparison(models_results, metric='mse', figsize=(10, 6), title=None):
        """
        Compare performance metrics across different models to identify the best approach.
        
        This plot provides a visual comparison of different models' performance using selected
        error metrics. It helps identify which model architecture or approach performs best for
        the neuronal forecasting task, supporting model selection decisions.
        
        Parameters:
        -----------
        models_results : dict
            Dictionary mapping {model_name: error_value}
        metric : str
            Metric name for the y-axis label
        figsize : tuple
            Figure size (width, height)
        title : str
            Plot title
        """
        plt.figure(figsize=figsize)
        
        # Sort models by performance (ascending for error metrics)
        sorted_models = sorted(models_results.items(), key=lambda x: x[1])
        model_names = [item[0] for item in sorted_models]
        metric_values = [item[1] for item in sorted_models]
        
        # Create bar plot
        bars = plt.bar(range(len(model_names)), metric_values, color='skyblue')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{metric_values[i]:.4f}',
                    ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.title(title or f'Model Comparison by {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    

    @staticmethod
    def plot_multiple_models(models_data, indices=None, num_series=40, time_steps=200, var_name='u', figsize=(15, 5)):
        """
        Plot multiple models side by side, each with its own Y-axis and selectable index for each model.
        
        This visualization allows direct comparison of different models' performance on the same or different
        neuron indices. It's useful for comparing how different model architectures capture the dynamics of
        specific neurons.
        
        Parameters:
        -----------
        models_data : dict
            Dictionary mapping {model_name: (original_data, forecasted_data)}
        indices : dict or None
            Dictionary mapping {model_name: index_to_plot}, if None uses index=0 for all
        num_series : int
            Maximum number of series to plot (if data is 2D)
        time_steps : int
            Number of time steps to show
        var_name : str
            Variable name (e.g., 'u' or 'v')
        figsize : tuple
            Figure size (width, height)
        """
        num_models = len(models_data)
        fig, axes = plt.subplots(1, num_models, figsize=figsize, sharex=True)  # Share only X axis

        if num_models == 1:
            axes = [axes]  # If there's only one model, transform axes into a list

        if indices is None:
            indices = {model_name: 0 for model_name in models_data}  # Default index=0 for all

        for ax, (model_name, (data, forecasted)) in zip(axes, models_data.items()):
            index = indices.get(model_name, 0)

