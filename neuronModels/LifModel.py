import numpy as np
from scipy.integrate import solve_ivp

class LifModel:
    def __init__(self, N, tau, R, I, V_rest=-65, V_th=-50, V_reset=-70):
        self.N = N  # Numero di neuroni
        self.tau = tau  # Costante di tempo
        self.R = R  # Resistenza di membrana
        self.I = I  # Corrente in ingresso
        self.V_rest = V_rest  # Potenziale di riposo
        self.V_th = V_th  # Soglia
        self.V_reset = V_reset  # Reset dopo lo spike
        self.spikes = [[] for _ in range(N)]  # Lista degli spike per ogni neurone

    def _model(self, t, V):
        """Equazione differenziale del modello LIF"""
        I_t = self.I(t) if callable(self.I) else self.I
        dVdt = (-(V - self.V_rest) + self.R * I_t) / self.tau
        return dVdt

    def simulate(self, T=100, dt=0.1):
        """Simulazione del modello LIF con reset individuale"""
        t_points = np.arange(0, T, dt)
        V_points = np.full((len(t_points), self.N), self.V_rest)

        V_current = np.full(self.N, self.V_rest)
        
        for i, t in enumerate(t_points[:-1]):
            V_next = V_current + dt * self._model(t, V_current)  # Eulero esplicito
            
            for j in range(self.N):
                if V_next[j] >= self.V_th:  # Controllo della soglia
                    self.spikes[j].append(t)
                    V_next[j] = self.V_reset  # Reset del solo neurone che ha sparato

            V_points[i + 1] = V_next
            V_current = V_next
        
        return t_points, V_points
