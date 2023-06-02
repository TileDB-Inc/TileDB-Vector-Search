from . import module as vs
from tiledbvspy import ColMajorMatrix_f32
import numpy as np

class FlatIndex:
    def __init__(self, uri):
        self.uri = uri
        self._index = None

        self._db = vs.load_as_matrix(uri)

    def query(self, targets: np.ndarray, k=10, nqueries=10, nthreads=8):
        # TODO:
        # - typecheck targets
        # - don't copy the array
        # - add all the options and query strategies

        assert targets.dtype == np.float32

        # TODO: make Matrix constructor from py::array. This is ugly (and copies).
        # Create a Matrix from the input targets
        targets_m = ColMajorMatrix_f32(*targets.shape)
        targets_m_a = np.array(targets_m, copy=False)
        targets_m_a[:] = targets

        r = vs.query_vq(
            self._db,
            targets_m,
            k,
            nqueries,
            nthreads
        )
        return np.array(r)