import numpy as np

from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.index import Index
from typing import Any, Mapping


class FlatIndex(Index):
    """
    Open a flat index

    Parameters
    ----------
    uri: str
        URI of the index
    config: Optional[Mapping[str, Any]]
        config dictionary, defaults to None
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(uri=uri, config=config)
        self.index_type = "FLAT"
        self._index = None
        self.db_uri = self.group[storage_formats[self.storage_version]["PARTS_ARRAY_NAME"] + self.index_version].uri
        schema = tiledb.ArraySchema.load(
            self.db_uri, ctx=tiledb.Ctx(self.config)
        )
        self.size = schema.domain.dim(1).domain[1]+1
        self._db = load_as_matrix(
            self.db_uri,
            ctx=self.ctx,
            config=config,
        )
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"] + self.index_version
        ].uri
        if tiledb.array_exists(self.ids_uri, self.ctx):
            self._ids = read_vector_u64(self.ctx, self.ids_uri, 0, 0)
        else:
            self._ids = StdVector_u64(np.arange(self.size).astype(np.uint64))

        dtype = self.group.meta.get("dtype", None)
        if dtype is None:
            self.dtype = self._db.dtype
        else:
            self.dtype = np.dtype(dtype)

    def query_internal(
        self,
        queries: np.ndarray,
        k: int = 10,
        nthreads: int = 8,
    ):
        """
        Query a flat index

        Parameters
        ----------
        queries: numpy.ndarray
            ND Array of queries
        k: int
            Number of top results to return per query
        nthreads: int
            Number of threads to use for query
        """
        # TODO:
        # - typecheck queries
        # - add all the options and query strategies

        assert queries.dtype == np.float32

        queries_m = array_to_matrix(np.transpose(queries))
        d, i = query_vq_heap(self._db, queries_m, self._ids, k, nthreads)

        return np.transpose(np.array(d)), np.transpose(np.array(i))
