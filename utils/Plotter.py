import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot_data(data=None, forecasted=None, index=0, num_series=5, time_steps=200, 
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