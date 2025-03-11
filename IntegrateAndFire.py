import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



class IntegrateAndFire:
    def __init__(self, N):
        self.N = N
        
    def _model(self,t, V, I=0.5, tau=10, V_rest=-65, V_thresh=-50, V_reset=-70):
        if V >= V_thresh:
            return V_reset - V  # Reset al potenziale di riposo dopo lo spike
        dVdt = (-(V - V_rest) + I) / tau
        return dVdt
