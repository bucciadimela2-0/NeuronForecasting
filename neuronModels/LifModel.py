import numpy as np

from neuronModels.PhysicsModels import PhysicsModels
from utils.Constants import Constants
from utils.Logger import Logger, LogLevel

logger = Logger()


class LifModel(PhysicsModels):

    TAU=10.0
    R=5.0
    I=10.0
    V_REST=-65        # Resting potential (mV)
    V_TH=-55          # Threshold potential (mV) - adjusted for more frequent spikes
    V_RESET=-75
    def __init__(self, N = Constants.N, tau = TAU, R=R, I=I, V_rest=V_REST, V_th=V_TH, V_reset=V_RESET):
        self.N = N  # Numero di neuroni
        self.tau = tau  # Costante di tempo
        self.R = R  # Resistenza di membrana
        self.I = I  # Corrente in ingresso
        self.V_rest = V_rest  # Potenziale di riposo
        self.V_th = V_th  # Soglia
        self.V_reset = V_reset  # Reset dopo lo spike
        self.spikes = [[] for _ in range(N)]  # Lista degli spike per ogni neurone
        logger.log("Instanitate LIF model", LogLevel.INFO)

    def _model(self, t, V):
       
        I_t = self.I(t) if callable(self.I) else self.I
        dVdt = (-(V - self.V_rest) + self.R * I_t) / self.tau
        return dVdt

    def _generate_synthetic_data(self, T=100, dt=0.1, use_saved_model = True):
        
        if use_saved_model:
            t_points, v_points = self._load_datas(Constants.LIF)
        else:
            t_points = np.arange(0, T, dt)
            v_points = np.full((len(t_points), self.N), self.V_rest)
            V_current = np.full(self.N, self.V_rest)
            
            for i, t in enumerate(t_points[:-1]):
                V_next = V_current + dt * self._model(t, V_current)  # Eulero esplicito
                
                for j in range(self.N):
                    if V_next[j] >= self.V_th:  # Controllo della soglia
                        self.spikes[j].append(t)
                        V_next[j] = self.V_reset  # Reset del solo neurone che ha sparato

                v_points[i + 1] = V_next
                V_current = V_next
            logger.log("Generated synthetic data for LIF model", LogLevel.INFO)
            self._save_datas(Constants.LIF, t_points,v_points)
        return t_points, v_points
