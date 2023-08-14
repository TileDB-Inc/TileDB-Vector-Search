import numpy as np

class Index:
    def query(self, targets: np.ndarray, k, **kwargs):
        raise NotImplementedError
