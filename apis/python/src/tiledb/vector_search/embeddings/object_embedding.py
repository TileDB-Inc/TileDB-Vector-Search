import numpy as np
from abc import ABC, abstractmethod 


class ObjectEmbedding(ABC):

    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def vector_type(self) -> np.dtype:
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def embed(self, objects, metadata=None) -> np.ndarray:
        raise NotImplementedError
