import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class izhikevich_model:
    def __init__(self, N, a, b, c, d, I, threshold=30):
        self.N = N  # Numero di neuroni
        self.a = a  # Parametro 'a'
        self.b = b  # Parametro 'b'
        self.c = c  # Parametro 'c'
        self.d = d  # Parametro 'd'
        self.I = I  # Corrente esterna
        self.threshold = threshold  # Soglia di attivazione del neurone
    
    def _model(self, t, y):
       
        N = len(y) // 2
        v = y[:N]  # Potenziale di membrana
        u = y[N:]  # Variabile di recupero

        dv = 0.04 * v**2 + 5 * v + 140 - u + self.I
        du = self.a * (self.b * v - u)

        # Condizione di "spike" e reset
        spike = v >= self.threshold
        v[spike] = self.c
        u[spike] += self.d

        return np.concatenate([dv, du])

    def _generate_synthetic_data(self, T=200, dt=0.1):
      
        # Condizioni iniziali
        y0 = np.random.uniform(-1, 1, self.N * 2)

        # Punti di valutazione nel tempo
        t_eval = np.arange(0, T, dt)

        # Risolvi il problema di valore iniziale
        sol = solve_ivp(
            self._model,
            [0, T],
            y0,
            method='RK45',
            t_eval=t_eval, vectorized=True
        )

        return sol.t, sol.y.T