import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot_comparison(y_test, forecasted, index=3, time_steps=200, 
                       var_name='u', title='Comparison Plot', figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.plot(y_test[:time_steps, index], label=f'Original {var_name}{index}')
        plt.plot(forecasted[:time_steps, index], 
                label=f'Corrected {var_name}{index}', 
                color='red', 
                linestyle='dashed')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
