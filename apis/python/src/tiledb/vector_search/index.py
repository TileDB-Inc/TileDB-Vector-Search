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
        # - add all the options and query strategies

        assert targets.dtype == np.float32

        targets_m = array_to_matrix(targets)

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
    memory_budget: int
        Main memory budget. If not provided no memory budget is applied.
    """

    def __init__(
        self, uri, dtype: np.dtype, memory_budget: int = -1, ctx: "Ctx" = None
    ):
        self.parts_db_uri = os.path.join(uri, "parts.tdb")
        self.centroids_uri = os.path.join(uri, "centroids.tdb")
        self.index_uri = os.path.join(uri, "index.tdb")
        self.ids_uri = os.path.join(uri, "ids.tdb")
        self.dtype = dtype
        self.memory_budget = memory_budget
        self.ctx = ctx
        if ctx is None:
            self.ctx = Ctx({})

        # TODO pass in a context
        if self.memory_budget == -1:
            self._db = load_as_matrix(self.parts_db_uri)
            self._ids = read_vector_u64(self.ctx, self.ids_uri)

        self._centroids = load_as_matrix(self.centroids_uri)
        self._index = read_vector_u64(self.ctx, self.index_uri)

    def query(
        self,
        targets: np.ndarray,
        k=10,
        nqueries=10,
        nthreads=8,
        nprobe=1,
        use_nuv_implementation: bool = False,
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
        use_nuv_implementation: bool
            wether to use the nuv query implementation. Default: False
        """
        assert targets.dtype == np.float32

        targets_m = array_to_matrix(targets)
        if self.memory_budget == -1:
            r = ivf_query_ram(
                self.dtype,
                self._db,
                self._centroids,
                targets_m,
                self._index,
                self._ids,
                nprobe=nprobe,
                k_nn=k,
                nth=True,  # ??
                nthreads=nthreads,
                ctx=self.ctx,
                use_nuv_implementation=use_nuv_implementation,
            )
        else:
            r = ivf_query(
                self.dtype,
                self.parts_db_uri,
                self._centroids,
                targets_m,
                self._index,
                self.ids_uri,
                nprobe=nprobe,
                k_nn=k,
                memory_budget=self.memory_budget,
                nth=True,  # ??
                nthreads=nthreads,
                ctx=self.ctx,
                use_nuv_implementation=use_nuv_implementation,
            )

        return np.array(r)
