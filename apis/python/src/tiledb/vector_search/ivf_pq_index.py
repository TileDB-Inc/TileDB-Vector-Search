"""
IVFPQ Index implementation.
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

INDEX_TYPE = "IVF_PQ"


class IVFPQIndex(index.Index):
    """
    Opens a `IVFPQIndex`.

    Parameters
    ----------
    uri: str
        URI of the index.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    timestamp: int or tuple(int)
        If int, open the index at a given timestamp.
        If tuple, open at the given start and end timestamps.
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
        self.index = vspy.IndexIVFPQ(self.ctx, uri, to_temporal_policy(timestamp))
        # TODO(paris): This is incorrect - should be fixed when we fix consolidation.
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
        nprobe: Optional[int] = 100,
        **kwargs,
    ):
        """
        Queries a `IVFPQIndex`.

        Parameters
        ----------
        queries: np.ndarray
            2D array of query vectors. This can be used as a batch query interface by passing multiple queries in one call.
        k: int
            Number of results to return per query vector.
        nprobe: int
            Number of partitions to check per query.
            Use this parameter to trade-off accuracy for latency and cost.
        """
        warnings.warn("The IVF PQ index is not yet supported, please use with caution.")
        if self.size == 0:
            return np.full((queries.shape[0], k), MAX_FLOAT32), np.full(
                (queries.shape[0], k), MAX_UINT64
            )

        if queries.ndim == 1:
            queries = np.array([queries])
        queries = np.transpose(queries)
        if not queries.flags.f_contiguous:
            queries = queries.copy(order="F")
        queries_feature_vector_array = vspy.FeatureVectorArray(queries)

        distances, ids = self.index.query(
            vspy.QueryType.InfiniteRAM, queries_feature_vector_array, k, nprobe
        )

        return np.array(distances, copy=False), np.array(ids, copy=False)


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    num_subspaces: int,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    partitions: Optional[int] = None,
    **kwargs,
) -> IVFPQIndex:
    """
    Creates an empty IVFPQIndex.
    Parameters
    ----------
    uri: str
        URI of the index.
    dimensions: int
        Number of dimensions for the vectors to be stored in the index.
    vector_type: np.dtype
        Datatype of vectors.
        Supported values (uint8, int8, float32).
    num_subspaces: int
        Number of subspaces to use in the PQ encoding. We will divide the dimensions into
        num_subspaces parts, and PQ encode each part separately. This means dimensions must
        be divisible by num_subspaces.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    storage_version: str
        The TileDB vector search storage version to use.
        If not provided, use the latest stable storage version.
    partitions: int
        Number of partitions to load the data with, if not provided, is auto-configured
        based on the dataset size.
    """
    warnings.warn("The IVF PQ index is not yet supported, please use with caution.")
    validate_storage_version(storage_version)
    # TODO(SC-49166): Support old storage versions with type-erased indexes.
    if storage_version == "0.1" or storage_version == "0.2":
        raise ValueError(
            f"Storage version {storage_version} is not supported for IVFPQIndex. IVFPQIndex requires storage version 0.3 or higher."
        )
    ctx = vspy.Ctx(config)
    if num_subspaces <= 0:
        raise ValueError(
            f"Number of num_subspaces ({num_subspaces}) must be greater than 0."
        )
    if dimensions % num_subspaces != 0:
        raise ValueError(
            f"Number of dimensions ({dimensions}) must be divisible by num_subspaces ({num_subspaces})."
        )
    index = vspy.IndexIVFPQ(
        feature_type=np.dtype(vector_type).name,
        id_type=np.dtype(np.uint64).name,
        partitioning_index_type=np.dtype(np.uint64).name,
        dimensions=dimensions,
        n_list=partitions if (partitions is not None and partitions is not -1) else 0,
        num_subspaces=num_subspaces,
    )
    # TODO(paris): Run all of this with a single C++ call.
    empty_vector = vspy.FeatureVectorArray(
        dimensions, 0, np.dtype(vector_type).name, np.dtype(np.uint64).name
    )
    index.train(empty_vector)
    index.add(empty_vector)
    index.write_index(ctx, uri, vspy.TemporalPolicy(0), storage_version)
    return IVFPQIndex(uri=uri, config=config)
