import numpy as np
from scipy.integrate import solve_ivp

from neuronModels.PhysicsModels import PhysicsModels
from utils.Constants import Constants
from utils.Logger import Logger, LogLevel

logger = Logger()


class izhikevich_model(PhysicsModels):
    N=20
    A=0.05
    B=0.2 
    C=-65
    D=8 
    I=10
    THRESHOLD = 30

    def __init__(self, N = Constants.N, a = A, b=B, c=C, d=D, I=I, threshold = THRESHOLD):
        self.N = N  # Numero di neuroni
        self.a = a  # Parametro 'a'
        self.b = b  # Parametro 'b'
        self.c = c  # Parametro 'c'
        self.d = d  # Parametro 'd'
        self.I = I  # Corrente esterna
        self.threshold = threshold  # Soglia di attivazione del neurone
        logger.log("Instanitate izhikevich model", LogLevel.INFO)

        
    
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

    def _generate_synthetic_data(self, T=200, dt=0.1, use_saved_models = True):
      
        if use_saved_models:
            t_points, v_points = self._load_datas(Constants.IZH)
        else: 
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
            t_points = sol.t
            v_points = sol.y.T
            logger.log("Generated synthetic data for izhikevich model", LogLevel.INFO)
            self._save_datas(Constants.IZH, t_points,v_points)

        return t_points,v_points