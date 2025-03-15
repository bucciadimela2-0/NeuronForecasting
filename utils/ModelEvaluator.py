import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


class ModelEvaluator:
    """
    Utility class for comprehensive evaluation of neural forecasting models.
    Includes cross-validation, multiple metrics, and visualization tools.
    """
    
    @staticmethod
    def evaluate_with_metrics(y_true, y_pred):
        """
        Calculate multiple evaluation metrics for model performance.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            Ground truth values
        y_pred : numpy.ndarray
            Predicted values
            
        Returns:
        --------
        dict
            Dictionary containing various performance metrics
        """
        # Ensure inputs are numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().numpy()
            
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate normalized RMSE (as percentage of data range)
        data_range = np.max(y_true) - np.min(y_true)
        nrmse = (rmse / data_range) * 100 if data_range > 0 else float('inf')
        
        # Calculate correlation coefficient (average across neurons if multi-dimensional)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            corr_coefs = []
            p_values = []
            for i in range(y_true.shape[1]):
                corr, p_val = pearsonr(y_true[:, i], y_pred[:, i])
                corr_coefs.append(corr)
                p_values.append(p_val)
            correlation = np.mean(corr_coefs)
            p_value = np.mean(p_values)
        else:
            correlation, p_value = pearsonr(y_true.flatten(), y_pred.flatten())
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nrmse_percent': nrmse,
            'correlation': correlation,
            'p_value': p_value
        }
    
    @staticmethod
    def cross_validate(model_class, X, y, n_splits=5, **model_params):
        """
        Perform k-fold cross-validation on a model.
        
        Parameters:
        -----------
        model_class : class
            The model class to instantiate
        X : numpy.ndarray or torch.Tensor
            Input features
        y : numpy.ndarray or torch.Tensor
            Target values
        n_splits : int
            Number of folds for cross-validation
        **model_params : dict
            Parameters to pass to the model constructor
            
        Returns:
        --------
        dict
            Dictionary containing cross-validation results
        """
        # Convert to torch tensors if they aren't already
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Initialize and train model
            model = model_class(**model_params)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            epochs = 100
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(X_train)
                loss = torch.nn.MSELoss()(outputs, y_train)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Evaluate on test fold
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                
            # Calculate metrics
            metrics = ModelEvaluator.evaluate_with_metrics(y_test.numpy(), y_pred.numpy())
            fold_metrics.append(metrics)
            
            # Store predictions for later analysis
            all_y_true.append(y_test.numpy())
            all_y_pred.append(y_pred.numpy())
            
        # Calculate average metrics across folds
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
            avg_metrics[f'{metric}_std'] = np.std([fold[metric] for fold in fold_metrics])
        
        # Combine all predictions
        all_y_true_combined = np.vstack(all_y_true)
        all_y_pred_combined = np.vstack(all_y_pred)
        
        return {
            'fold_metrics': fold_metrics,
            'avg_metrics': avg_metrics,
            'all_y_true': all_y_true_combined,
            'all_y_pred': all_y_pred_combined
        }
    
    @staticmethod
    def plot_prediction_quality(y_true, y_pred, title="Prediction Quality", figsize=(10, 8)):
        """
        Create diagnostic plots to assess prediction quality.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            Ground truth values
        y_pred : numpy.ndarray
            Predicted values
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        # Ensure inputs are numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().numpy()
            
        # Flatten arrays if they're multi-dimensional
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # 1. Scatter plot of predicted vs true values
        axs[0, 0].scatter(y_true_flat, y_pred_flat, alpha=0.5)
        axs[0, 0].plot([y_true_flat.min(), y_true_flat.max()], 
                      [y_true_flat.min(), y_true_flat.max()], 'r--')
        axs[0, 0].set_xlabel('True Values')
        axs[0, 0].set_ylabel('Predicted Values')
        axs[0, 0].set_title('Prediction vs Ground Truth')
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 2. Histogram of residuals
        residuals = y_pred_flat - y_true_flat
        axs[0, 1].hist(residuals, bins=30, alpha=0.7, color='blue')
        axs[0, 1].axvline(x=0, color='r', linestyle='--')
        axs[0, 1].set_xlabel('Residuals')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Residual Distribution')
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 3. Residuals vs predicted values (to check for heteroscedasticity)
        axs[1, 0].scatter(y_pred_flat, residuals, alpha=0.5)
        axs[1, 0].axhline(y=0, color='r', linestyle='--')
        axs[1, 0].set_xlabel('Predicted Values')
        axs[1, 0].set_ylabel('Residuals')
        axs[1, 0].set_title('Residuals vs Predicted')
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 4. Q-Q plot of residuals (to check for normality)
        from scipy import stats
        stats.probplot(residuals, plot=axs[1, 1])
        axs[1, 1].set_title('Q-Q Plot of Residuals')
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        plt.show()
        
        # Print summary statistics
        metrics = ModelEvaluator.evaluate_with_metrics(y_true, y_pred)
        print(f"\nPerformance Metrics:")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"R²: {metrics['r2']:.6f}")
        print(f"NRMSE: {metrics['nrmse_percent']:.2f}%")
        print(f"Correlation: {metrics['correlation']:.6f} (p-value: {metrics['p_value']:.6f})")
        
        return metrics