from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, List

class BaseIntentModel(ABC):
    @abstractmethod
    def create_model(self):
        pass
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass
        
    @abstractmethod
    def save_model(self, path: str):
        pass
        
    @abstractmethod
    def load_model(self, path: str):
        pass