
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class fitzhughNagumo:
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


        

