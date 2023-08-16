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
        URI of datataset
    dtype: numpy.dtype
        datatype float32 or uint8
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(uri=uri, config=config)
        self._index = None
        self.index_uri = self.group[storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]].uri
        self._db = load_as_matrix(
            self.index_uri,
            ctx=self.ctx,
            config=config,
        )
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"]
        ].uri
        if tiledb.array_exists(self.ids_uri, self.ctx):
            self._ids = read_vector_u64(self.ctx, self.ids_uri, 0, 0)
        else:
            schema = tiledb.ArraySchema.load(
                self.index_uri, ctx=tiledb.Ctx(self.config)
            )
            self.size = schema.domain.dim(1).domain[1]
            self._ids = StdVector_u64(np.arange(self.size).astype(np.uint64))

        dtype = self.group.meta.get("dtype", None)
        if dtype is None:
            self.dtype = self._db.dtype
        else:
            self.dtype = np.dtype(dtype)

    def query_internal(
        self,
        targets: np.ndarray,
        k: int = 10,
        nthreads: int = 8,
    ):
        """
        Query a flat index

        Parameters
        ----------
        targets: numpy.ndarray
            ND Array of query targets
        k: int
            Number of top results to return per target
        nqueries: int
            Number of queries
        nthreads: int
            Number of threads to use for query
        """
        # TODO:
        # - typecheck targets
        # - add all the options and query strategies

        assert targets.dtype == np.float32

        targets_m = array_to_matrix(np.transpose(targets))
        r = query_vq_heap(self._db, targets_m, self._ids, k, nthreads)

        return np.transpose(np.array(r))