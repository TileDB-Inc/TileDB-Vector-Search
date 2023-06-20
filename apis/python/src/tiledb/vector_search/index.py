import os

import numpy as np

from tiledb.vector_search.module import *


class Index:
    def query(self, targets: np.ndarray, k=10, nqueries=10, nthreads=8, nprobe=1):
        raise NotImplementedError


class FlatIndex(Index):
    """
    Open a flat index

    Parameters
    ----------
    uri: str
        URI of datataset
    dtype: numpy.dtype
        datatype float32 or uint8
    parts_name: str
        Optional name of partitions
    """

    def __init__(self, uri: str, dtype: np.dtype, parts_name: str = "parts.tdb"):
        self.uri = uri
        self.dtype = dtype
        self._index = None

        self._db = load_as_matrix(os.path.join(uri, parts_name))

    def query(
        self,
        targets: np.ndarray,
        k: int = 10,
        nqueries: int = 10,
        nthreads: int = 8,
        nprobe: int = 1,
    ):
        """
        Open a flat index

        Parameters
        ----------
        targets: numpy.ndarray
            ND Array of query targets
        k: int
            Number of top results to return per target
        nqueries: int
            Number of queries
        nthreads: int
            Number of threads to use for queyr
        nprobe: int
            number of probes
        """
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

        r = query_vq(self._db, targets_m, k, nqueries, nthreads)
        return np.array(r)


class IVFFlatIndex(Index):
    """
    Open a IVF Flat index

    Parameters
    ----------
    uri: str
        URI of datataset
    dtype: numpy.dtype
        datatype float32 or uint8
    """

    def __init__(self, uri, dtype: np.dtype):
        self.parts_db_uri = os.path.join(uri, "parts.tdb")
        self.centroids_uri = os.path.join(uri, "centroids.tdb")
        self.index_uri = os.path.join(uri, "index.tdb")
        self.ids_uri = os.path.join(uri, "ids.tdb")
        self.dtype = dtype

        ctx = Ctx({})  # TODO pass in a context
        self._db = load_as_matrix(self.parts_db_uri)
        self._centroids = load_as_matrix(self.centroids_uri)
        self._index = read_vector_u64(ctx, self.index_uri)
        self._ids = read_vector_u64(ctx, self.ids_uri)

    def query(self, targets: np.ndarray, k=10, nqueries=10, nthreads=8, nprobe=1):
        """
        Open a flat index

        Parameters
        ----------
        targets: numpy.ndarray
            ND Array of query targets
        k: int
            Number of top results to return per target
        nqueries: int
            Number of queries
        nthreads: int
            Number of threads to use for queyr
        nprobe: int
            number of probes
        """
        assert targets.dtype == np.float32

        # TODO: use Matrix constructor from py::array
        targets_m = ColMajorMatrix_f32(*targets.shape)
        targets_m_a = np.array(targets_m, copy=False)
        targets_m_a[:] = targets

        r = query_kmeans(
            self._db.dtype,
            self._db,
            self._centroids,
            targets_m,
            self._index,
            self._ids,
            nprobe=nprobe,
            k_nn=k,
            nth=True,  # ??
            nthreads=nthreads,
        )
        return np.array(r)
