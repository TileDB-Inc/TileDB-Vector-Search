"""
Vamana Index implementation.

Vamana is based on Microsoft's DiskANN vector search library, as described in these papers:
```
  Subramanya, Suhas Jayaram, and Rohan Kadekodi. DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node.

  Singh, Aditi, et al. FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search. arXiv:2105.09613, arXiv, 20 May 2021, http://arxiv.org/abs/2105.09613.

  Gollapudi, Siddharth, et al. “Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters.” Proceedings of the ACM Web Conference 2023, ACM, 2023, pp. 3406-16, https://doi.org/10.1145/3543507.3583552.
```
"""
import warnings
from typing import Any, Mapping

import numpy as np

from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search import index
from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import STORAGE_VERSION
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.storage_formats import validate_storage_version
from tiledb.vector_search.utils import MAX_FLOAT32
from tiledb.vector_search.utils import MAX_UINT64
from tiledb.vector_search.utils import to_temporal_policy

INDEX_TYPE = "VAMANA"

L_BUILD_DEFAULT = 100
R_MAX_DEGREE_DEFAULT = 64
L_SEARCH_DEFAULT = 100


class VamanaIndex(index.Index):
    """
    Opens a `VamanaIndex`.

    Parameters
    ----------
    uri: str
        URI of the index.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    open_for_remote_query_execution: bool
        If `True`, do not load any index data in main memory locally, and instead load index data in the TileDB Cloud taskgraph created when a non-`None` `driver_mode` is passed to `query()`.
        If `False`, load index data in main memory locally. Note that you can still use a taskgraph for query execution, you'll just end up loading the data both on your local machine and in the cloud taskgraph.
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        open_for_remote_query_execution: bool = False,
        **kwargs,
    ):
        self.index_open_kwargs = {
            "uri": uri,
            "config": config,
            "timestamp": timestamp,
        }
        self.index_open_kwargs.update(kwargs)
        self.index_type = INDEX_TYPE
        super().__init__(
            uri=uri,
            config=config,
            timestamp=timestamp,
            open_for_remote_query_execution=open_for_remote_query_execution,
        )
        # TODO(SC-48710): Add support for `open_for_remote_query_execution`. We don't leave `self.index`` as `None` because we need to be able to call index.dimensions().
        self.index = vspy.IndexVamana(self.ctx, uri, to_temporal_policy(timestamp))
        self.db_uri = self.group[
            storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]
        ].uri
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"]
        ].uri

        schema = tiledb.ArraySchema.load(self.db_uri, ctx=tiledb.Ctx(self.config))
        self.dimensions = self.index.dimensions()

        self.dtype = np.dtype(self.group.meta.get("dtype", None))
        if self.dtype is None:
            self.dtype = np.dtype(schema.attr("values").dtype)
        else:
            self.dtype = np.dtype(self.dtype)

        if self.base_size == -1:
            self.size = schema.domain.dim(1).domain[1] + 1
        else:
            self.size = self.base_size

    def get_dimensions(self):
        """
        Returns the dimension of the vectors in the index.
        """
        return self.dimensions

    def query_internal(
        self,
        queries: np.ndarray,
        k: int = 10,
        l_search: Optional[int] = L_SEARCH_DEFAULT,
        **kwargs,
    ):
        """
        Queries a `VamanaIndex`.

        Parameters
        ----------
        queries: np.ndarray
            2D array of query vectors. This can be used as a batch query interface by passing multiple queries in one call.
        k: int
            Number of results to return per query vector.
        l_search: int
            How deep to search. Larger parameters will result in slower latencies, but higher accuracies.
            Should be >= k, and if it's not, we will set it to k.
        """
        if self.size == 0:
            return np.full((queries.shape[0], k), MAX_FLOAT32), np.full(
                (queries.shape[0], k), MAX_UINT64
            )

        if l_search < k:
            warnings.warn(f"l_search ({l_search}) should be >= k ({k}), setting to k")
            l_search = k

        if queries.ndim == 1:
            queries = np.array([queries])
        queries = np.transpose(queries)
        if not queries.flags.f_contiguous:
            queries = queries.copy(order="F")
        queries_feature_vector_array = vspy.FeatureVectorArray(queries)

        distances, ids = self.index.query(queries_feature_vector_array, k, l_search)

        return np.array(distances, copy=False), np.array(ids, copy=False)


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    l_build: int = L_BUILD_DEFAULT,
    r_max_degree: int = R_MAX_DEGREE_DEFAULT,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    **kwargs,
) -> VamanaIndex:
    """
    Creates an empty VamanaIndex.
    Parameters
    ----------
    uri: str
        URI of the index.
    dimensions: int
        Number of dimensions for the vectors to be stored in the index.
    vector_type: np.dtype
        Datatype of vectors.
        Supported values (uint8, int8, float32).
    l_build: int
        The number of neighbors considered for each node during construction of the graph. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. l_build should be >= r_max_degree unless you need to build indices quickly and can compromise on quality.
        Typically between 75 and 200. If not provided, use the default value of 100.
    r_max_degree: int
        The maximum degree for each node in the final graph. Larger values will result in larger indices and longer indexing times, but better search quality.
        Typically between 60 and 150. If not provided, use the default value of 64.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    storage_version: str
        The TileDB vector search storage version to use.
        If not provided, use the latest stable storage version.
    """
    validate_storage_version(storage_version)
    # TODO(SC-49166): Support old storage versions with type-erased indexes.
    if storage_version == "0.1" or storage_version == "0.2":
        raise ValueError(
            f"Storage version {storage_version} is not supported for VamanaIndex. VamanaIndex requires storage version 0.3 or higher."
        )
    ctx = vspy.Ctx(config)
    index = vspy.IndexVamana(
        feature_type=np.dtype(vector_type).name,
        id_type=np.dtype(np.uint64).name,
        dimensions=dimensions,
        l_build=l_build if l_build > 0 else L_BUILD_DEFAULT,
        r_max_degree=r_max_degree if l_build > 0 else R_MAX_DEGREE_DEFAULT,
    )
    # TODO(paris): Run all of this with a single C++ call.
    empty_vector = vspy.FeatureVectorArray(
        dimensions, 0, np.dtype(vector_type).name, np.dtype(np.uint64).name
    )
    index.train(empty_vector)
    index.add(empty_vector)
    index.write_index(ctx, uri, vspy.TemporalPolicy(0), storage_version)
    return VamanaIndex(uri=uri, config=config)
