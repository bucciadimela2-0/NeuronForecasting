import numpy as np
from scipy.integrate import solve_ivp

from neuronModels.PhysicsModels import PhysicsModels
from utils.Constants import Constants
from utils.DataHandler import DataHandler
from utils.Logger import Logger, LogLevel
import matplotlib.pyplot as plt

logger = Logger()


class FitzhughNagumoModel(PhysicsModels):
    EPSILON = 0.05
    SIGMA = 0.2  # Coupling strength
    A = 0.5  # Threshold parameter
    PHI = np.pi / 2 + 0.1
    B = np.array([[np.cos(PHI), np.sin(PHI)], [-np.sin(PHI), np.cos(PHI)]])
    G = DataHandler.generate_adjacency_matrix(Constants.A1, n=3)

    def __init__(self, N = Constants.N ,epsilon = EPSILON ,sigma = SIGMA, a = A, B = B, G = G):
        super(PhysicsModels).__init__
        self.N = N
        self.sigma = sigma
        self.epsilon = epsilon
        self.a = a
        self.B = B
        self.G = G
        logger.log("Instanitate Fitzhugh-Nagumo model",  LogLevel.INFO)

    def _model(self,t,y):
        
        N = len(y) // 2
        u = y[:N]
        v = y[N:]
        du = np.zeros(N)
        dv = np.zeros(N)
        for k in range(N):
            coupling_u = self.sigma * np.sum(self.G[k, :] * (self.B[0, 0] * u + self.B[0, 1] * v))
            coupling_v = self.sigma * np.sum(self.G[k, :] * (self.B[1, 0] * u + self.B[1, 1] * v))
            du[k] = (u[k] - u[k]**3 / 3 - v[k] + coupling_u) / self.epsilon
            dv[k] = u[k] + self.a + coupling_v
        return np.concatenate([du, dv])
    
    def _generate_synthetic_data(self, T=100, dt=0.1, use_saved_models = True):
        """
        Generate training data using physical model
        """
        
        if use_saved_models:
            t_points, v_points = self._load_datas(Constants.FN)
        else: 

            y0 = np.random.uniform(-1, 1, self.N * 2)


            
            t_eval = np.arange(0, T, dt)

          
            sol = solve_ivp(
                self._model,
                [0,T],
                y0,
                method='RK45',
                t_eval=t_eval, vectorized= True
            )
            t_points = sol.t
            v_points = sol.y.T

            logger.log("Generated synthetic data for FitzHugh-Nagumo model", LogLevel.INFO)
            self._save_datas(Constants.FN, t_points,v_points)

        return t_points, v_points


    

