import numpy as np
from scipy.integrate import solve_ivp

from neuronModels.PhysicsModels import PhysicsModels


class FitzhughNagumoModel(PhysicsModels):
    def __init__(self,N,epsilon,sigma, a, B, G):
        self.N = N
        self.sigma = sigma
        self.epsilon = epsilon
        self.a = a
        self.B = B
        self.G = G

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
    
    #ha senso averlo qui? o lo metto nella classe specifica?
    def _generate_synthetic_data(self, T=100, dt=0.1):
        """
        Generate training data using physical model
        """
        # Initial conditions
        #mhm...
        y0 = np.random.uniform(-1, 1, self.N * 2)


        # Time evaluation points
        t_eval = np.arange(0, T, dt)

        # Solve the initial value problem
        sol = solve_ivp(
            self._model,
            [0,T],
            y0,
            method='RK45',
            t_eval=t_eval, vectorized= True
        )

        return sol.t, sol.y.T


        

