from abc import ABC, abstractmethod


class PhysicsModels(ABC): 

    def __init__(self):
        super().__init__() 
    
    @abstractmethod
    def _generate_synthetic_data(self, T=100, dt=0.1):
        pass

    @abstractmethod
    def _model(self,t,y): 
        pass
