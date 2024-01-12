import numpy as np


class ObjectEmbedding:
    def dimensions(self) -> int:
        raise NotImplementedError

    def vector_type(self) -> np.dtype:
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def embed(self, objects, metadata=None) -> np.ndarray:
        raise NotImplementedError
