import os

import numpy as np
from tiledb.vector_search import _tiledbvspy as vspy

from . import module as vs
from ._tiledbvspy import ColMajorMatrix_f32, read_vector_u64


class Index:
    def query(self, targets: np.ndarray, k=10, nqueries=10, nthreads=8):
        raise NotImplementedError


class FlatIndex(Index):
    def __init__(self, uri):
        self.uri = uri
        self._index = None

        self._db = vs.load_as_matrix(os.path.join(uri, "parts.tdb"))

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

        r = vs.query_vq(self._db, targets_m, k, nqueries, nthreads)
        return np.array(r)


class IVFFlatIndex(Index):
    def __init__(self, uri):
        self.parts_db_uri = os.path.join(uri, "parts.tdb")
        self.centroids_uri = os.path.join(uri, "centroids.tdb")
        self.index_uri = os.path.join(uri, "index.tdb")
        self.ids_uri = os.path.join(uri, "ids.tdb")

        ctx = vspy.Ctx({})  # TODO pass in a context
        # TODO self._db = vs.load_as_matrix(self.db_uri)
        self._centroids = vs.load_as_matrix(self.centroids_uri)
        self._index = read_vector_u64(ctx, self.index_uri)
        # self._ids = vs.load_as_matrix(self.ids_uri)

    def query(self, targets: np.ndarray, k=10, nqueries=10, nthreads=8):
        assert targets.dtype == np.float32

        # TODO: use Matrix constructor from py::array
        targets_m = ColMajorMatrix_f32(*targets.shape)
        targets_m_a = np.array(targets_m, copy=False)
        targets_m_a[:] = targets

        r = vs.query_kmeans(
            self.parts_db_uri,
            self._centroids,
            targets_m,
            self._index,
            self.ids_uri,
            k,
            nqueries,
            True,  # ??
            nthreads,
        )
        result = np.array(r)
        return result
