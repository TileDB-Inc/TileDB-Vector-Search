import numpy as np

from ._tiledbvspy import ColMajorMatrix_f32
from . import module as vs


class FlatIndex:
    def __init__(self, uri):
        self.uri = uri
        self._index = None

        self._db = vs.load_as_matrix(uri)

    def query(self, targets: np.ndarray, k=10, nthreads=8):
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

        r = vs.query_vq(self._db, targets_m, k, nthreads)
        return np.array(r)


class KMeansIndex:
    # TODO: this should support a Group URI
    def __init__(self, db_uri, centroids_uri, index_uri, ids_uri):
        self.db_uri = db_uri
        self.centroids_uri = centroids_uri
        self.index_uri = index_uri
        self.ids_uri = ids_uri

        # TODO self._db = vs.load_as_matrix(self.db_uri)
        self._centroids = vs.load_as_matrix(self.centroids_uri)
        self._index = vs.load_as_matrix(self.index_uri)
        # self._ids = vs.load_as_matrix(self.ids_uri)

    def query(self, targets: np.ndarray, k=10, nqueries=10, nthreads=8):
        assert targets.dtype == np.float32

        # TODO: use Matrix constructor from py::array
        targets_m = ColMajorMatrix_f32(*targets.shape)
        targets_m_a = np.array(targets_m, copy=False)
        targets_m_a[:] = targets

        r = vs.query_vq(
            self.db_uri,
            self._centroids,
            targets_m,
            self._index,
            self.ids_uri,
            k,
            nqueries,
            nthreads,
        )
        return np.array(r)
