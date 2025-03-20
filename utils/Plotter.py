import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
import os
from utils.DataHandler import DataHandler
from utils.Constants import Constants


class Plotter:

   

    @staticmethod
    def plot_data(data=None, forecasted=None, index=26, num_series=1, time_steps=100, 
                  var_name='u', title=None, figsize=(10, 5), y_test=None, color_palette=None, model_name = None):
        
        if color_palette is None:
            color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
       
        plt.style.use('seaborn-whitegrid')  
        fig = plt.figure(figsize=figsize)

        
        if y_test is not None:
            
            plt.plot(y_test[:time_steps, index], 
                     label=f'Original {var_name}{index}', color='blue', linewidth=1.5, alpha=0.9)
            if forecasted is not None:
                plt.plot(forecasted[:time_steps, index], 
                         label=f'Forecasted {var_name}{index}', color='red', linestyle='-', linewidth=1.5, alpha=0.8)
        else:
            
            n_series = min(num_series, data.shape[1]) if len(data.shape) > 1 else 1
            if forecasted is not None:
                for i in range(n_series):
                    plt.plot(data[:time_steps, index], 
                             label=f'Original {var_name}{index}', color=color_palette[0], linewidth=1.5, alpha=0.9)
                    plt.plot(forecasted[:time_steps, i], 
                             label=f'Forecasted {var_name}{index}', color='red', linestyle='-', linewidth=1.5, alpha=0.8)
            else:
                for i in range(n_series):
                    plt.plot(data[:time_steps, index], 
                             label=f'Original {var_name}{index}', color=color_palette[0], linewidth=1.5, alpha=0.9)

       
        plt.title(title or 'Time Series Comparison', fontsize=14, fontweight='light', color='black')
        plt.xlabel('Time Steps', fontsize=12, color='black')
        plt.ylabel('Amplitude', fontsize=12, color='black')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=False)  
        plt.grid(True, linestyle='-', alpha=0.4, color='gray')
        plt.tight_layout()
        plt.gca().set_facecolor('#f9f9f9')
        plt.xticks(color='black')
        plt.yticks(color='black')
        DataHandler.save_plot(fig, f"TimeSeriesComparison_{model_name}", output_dir="figures", format="png")



    @staticmethod
    def plot_phase_space(data, forecasted=None, time_steps=100, var_indices=(0, 20), 
                         figsize=(8, 6), title="Phase Space Representation"):
        
        data = data[:time_steps]
        if forecasted is not None:
            forecasted = forecasted[:time_steps]
        
        n_vars = data.shape[1]
        dim = len(var_indices) if n_vars > 1 else 2  # 2D if only one variable

        # Compute derivatives for single-variable case (x, dx/dt)
        if n_vars == 1:
            dx_dt = np.gradient(data[:, 0])
            plt.figure(figsize=figsize)
            plt.plot(data[:, 0], dx_dt, label="Original", color="black", linewidth=1.5)
            if forecasted is not None:
                dx_dt_forecasted = np.gradient(forecasted[:, 0])
                plt.plot(forecasted[:, 0], dx_dt_forecasted, label="Forecasted", linestyle="dashed", color="red")
            plt.xlabel("x")
            plt.ylabel("dx/dt")
            plt.title(title)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()
        
        # Multi-variable case (2D or 3D phase space)
        else:
            fig = plt.figure(figsize=figsize)
            
            if dim == 2:
                plt.plot(data[:, var_indices[0]], data[:, var_indices[1]], label="Original", color="black", linewidth=1.5)
                if forecasted is not None:
                    plt.plot(forecasted[:, var_indices[0]], forecasted[:, var_indices[1]], 
                             label="Forecasted", linestyle="dashed", color="red")
                plt.xlabel(f"Variable {var_indices[0]}")
                plt.ylabel(f"Variable {var_indices[1]}")
            else:
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(data[:, var_indices[0]], data[:, var_indices[1]], data[:, var_indices[2]], 
                        label="Original", color="black", linewidth=1.5)
                if forecasted is not None:
                    ax.plot(forecasted[:, var_indices[0]], forecasted[:, var_indices[1]], forecasted[:, var_indices[2]], 
                            label="Forecasted", linestyle="dashed", color="red")
                ax.set_xlabel(f"Variable {var_indices[0]}")
                ax.set_ylabel(f"Variable {var_indices[1]}")
                ax.set_zlabel(f"Variable {var_indices[2]}")
            
            plt.title(title)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            DataHandler.save_plot(fig, "PhaseSpace", output_dir="figures", format="png")


    
    @staticmethod
    def plot_results_chimera_analysis(t_points, v_points):
        
        N = Constants.N
        u = v_points[:, :N]  # Estraggo solo le variabili u
        v = v_points[:, N:]  # Estraggo solo le variabili v

        # Spazio-tempo di u_k
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(u, aspect='auto', cmap='plasma', extent=[0, N, 0,14])
        plt.colorbar(label="$u_k$")
        plt.ylabel("Tempo")
        plt.xlabel("Indice oscillatore")
        plt.title("Spazio-tempo di $u_k$")
        DataHandler.save_plot(fig, "SpazioTempouk", output_dir="figures", format="png")
       
        # Velocità di fase media ω_k
        dt = np.gradient(t_points) 
        du = np.gradient(u, axis=0)  
        omega_k = np.mean(du / dt[:, None], axis=0) 

        fig = plt.figure(figsize=(6, 4))
        plt.plot(range(N), omega_k, 'o-', label=r'$\omega_k$')
        plt.axhline(y=np.mean(omega_k), color='r', linestyle='--', label=r'Media $\omega_k$')
        plt.xlabel("Indice oscillatore")
        plt.ylabel("Velocità di fase media")
        plt.title("Velocità di fase media degli oscillatori")
        plt.legend()
        DataHandler.save_plot(fig, "MeanVelocity", output_dir="figures", format="png")

        # Parametro di ordine di Kuramoto
        phases = np.arctan2(v, u)  # Angolo di fase
        kuramoto_order = np.abs(np.mean(np.exp(1j * phases), axis=1))

        fig = plt.figure(figsize=(6, 4))
        plt.plot(t_points, kuramoto_order, label="$r(t)$", color="b")
        plt.xlabel("Tempo")
        plt.ylabel("Parametro di ordine")
        plt.title("Dinamica del parametro di ordine di Kuramoto")
        plt.legend()
        DataHandler.save_plot(fig, "Kuramoto", output_dir="figures", format="png")

     
   
    @staticmethod

    def plot_phase_distribution(self,phases, time_idx=-1, figsize=(10, 6)):
       
        N = phases.shape[1]
        phases_t = phases[time_idx, :]
        
        # Plot circolare delle fasi
        fig = plt.figure(figsize=figsize)
        
        # Conversione a coordinate cartesiane
        x = np.cos(phases_t)
        y = np.sin(phases_t)
        
        # Plot dei punti sul cerchio unitario
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        plt.gca().add_patch(circle)
        plt.scatter(x, y, c=range(N), cmap='viridis')
        plt.colorbar(label='Indice oscillatore')
        
        # Aggiungi linee dal centro ai punti per visualizzare meglio le fasi
        for i in range(N):
            plt.plot([0, x[i]], [0, y[i]], 'gray', alpha=0.3)
        
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.gca().set_aspect('equal')
        plt.title('Distribuzione delle fasi degli oscillatori')
        plt.grid(True)
        DataHandler.save_plot(fig, "PhaseDistribution", output_dir="figures", format="png")

   
    def plot_synchronization_map(phases, t_points, window_size=10, figsize=(12, 6)):
       
        N = phases.shape[1]
        T = phases.shape[0]
        
        # Calcolo del parametro d'ordine locale per ogni oscillatore e istante temporale
        sync_map = np.zeros((T - window_size, N))
        
        for t in range(T - window_size):
            for i in range(N):
                
                phase_window = phases[t:t+window_size, :]
                neighbors = [(i + j) % N for j in range(-2, 3)]  
                complex_order = np.mean(np.exp(1j * phase_window[:, neighbors]), axis=1)
                sync_map[t, i] = np.mean(np.abs(complex_order))
        
       
        fig = plt.figure(figsize=figsize)
        plt.imshow(sync_map.T, aspect='auto', origin='lower', 
                extent=[t_points[0], t_points[-window_size], 0, N-1], 
                cmap='viridis')
        plt.colorbar(label='Grado di sincronizzazione locale')
        plt.xlabel('Tempo')
        plt.ylabel('Indice oscillatore')
        plt.title('Mappa di sincronizzazione spazio-temporale')
        plt.tight_layout()
        DataHandler.save_plot(fig, "SyncMap", output_dir="figures", format="png")
        
        return sync_map

    @staticmethod
    def plot_forecast(t_points, noisy_data, physical_predictions, hybrid_predictions, assimilation_window, model_name = None, neuron_idx = 0, output_dir = 'figures'):
       
        
        # Plot time series
        fig = plt.subplot(2, 1, 1)
        plt.plot(t_points[-len(noisy_data):], noisy_data[:, neuron_idx], 'k-', label='True Values')
        plt.plot(t_points[-len(physical_predictions):], physical_predictions[:, neuron_idx], 'b--', label='Physical Model')
        plt.plot(t_points[-len(hybrid_predictions):], hybrid_predictions[:, neuron_idx], 'r-', label='Hybrid Model')

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
        
        if assimilation_window > 0:
            plt.axvline(x=assimilation_time, color='g', linestyle='--', label='End of Assimilation')

        figure_path = os.path.join(output_dir,model_name + 'hybrid_model_forecast.png')
        plt.title('Absolute Error')
        plt.xlabel('Time')
        plt.ylabel('Error Magnitude')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(figure_path)
        
    
       
        
       





    