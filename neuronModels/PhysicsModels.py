import os
from abc import ABC, abstractmethod

import numpy as np

from utils.Logger import Logger, LogLevel

logger = Logger()

class PhysicsModels(ABC): 

    def __init__(self):
        super().__init__() 
    
    @abstractmethod
    def _generate_synthetic_data(self, T=100, dt=0.1):
        pass

    @abstractmethod
    def _model(self,t,y): 
        pass

    

    def _save_datas(self, model_name, t_points, v_points):
        MODEL_PATH = os.path.join("neuronModels", "SavedModels")
        
        # Creare la directory se non esiste
        os.makedirs(MODEL_PATH, exist_ok=True)
        file_path = os.path.join(MODEL_PATH, model_name + ".npz")
        logger.log(f"Saving data to {file_path}", LogLevel.INFO)
        np.savez(file_path, t_points=t_points, v_points=v_points)

    def _load_datas(self, model_name):
        MODEL_PATH = os.path.join("neuronModels", "SavedModels")
        
        
        file_path = os.path.join(MODEL_PATH, model_name + ".npz")

    
        if not os.path.exists(file_path):
            logger.log(f"File {file_path} not found!", LogLevel.ERROR)
            raise FileNotFoundError(f"File {file_path} not found!")

        
        data = np.load(file_path)

        
        t_points = data["t_points"]
        v_points = data["v_points"]

        logger.log(f"Loaded data from {file_path}", LogLevel.INFO)

        return t_points, v_points



        
        

