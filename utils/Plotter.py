import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def plot_data(data=None, forecasted=None, index=0, num_series=5, time_steps=100, 
                  var_name='u', title=None, figsize=(10, 5), y_test=None):
        """
        Unified plotting method for time series data
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
    def plot_multiple_models(models_data, indices=None, num_series=40, time_steps=200, var_name='u', figsize=(15, 5)):
        """
        Plotta più modelli affiancati, ciascuno con il proprio asse Y e con un indice selezionabile per ogni modello.
        
        :param models_data: Dizionario {nome_modello: (data, forecasted)}
        :param indices: Dizionario {nome_modello: indice_da_plottare}, se None prende index=0
        :param num_series: Numero massimo di serie da plottare (se i dati sono 2D)
        :param time_steps: Numero di passi temporali da mostrare
        :param var_name: Nome della variabile (es. 'u' o 'v')
        :param figsize: Dimensione della figura (larghezza, altezza)
        """
        num_models = len(models_data)
        fig, axes = plt.subplots(1, num_models, figsize=figsize, sharex=True)  # Condivide solo asse X

        if num_models == 1:
            axes = [axes]  # Se c'è un solo modello, trasformiamo axes in lista

        if indices is None:
            indices = {model_name: 0 for model_name in models_data}  # Default index=0 per tutti

        for ax, (model_name, (data, forecasted)) in zip(axes, models_data.items()):
            index = indices.get(model_name, 0)  # Prende l'indice specifico o usa 0 di default

            if data.ndim == 1:  # Caso 1D (solo una serie)
                ax.plot(data[:time_steps], label=f'Original {var_name}')
                if forecasted is not None:
                    ax.plot(forecasted[:time_steps], label=f'Forecasted {var_name}', linestyle='dashed', color='red')
            else:  # Caso 2D (più serie)
                n_series = min(num_series, data.shape[1])
                if index >= n_series:
                    raise ValueError(f"Indice {index} fuori dal range (max {n_series-1}) per il modello {model_name}")

                ax.plot(data[:time_steps, index], label=f'Original {var_name}{index}')
                if forecasted is not None:
                    ax.plot(forecasted[:time_steps, index], label=f'Forecasted {var_name}{index}', linestyle='dashed', color='red')

            ax.set_title(f'{model_name} (Index {index})')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

