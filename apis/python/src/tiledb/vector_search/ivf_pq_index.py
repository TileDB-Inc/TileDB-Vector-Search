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
    memory_budget: int
        Main memory budget, in number of vectors, for query execution.
        If not provided, all index data are loaded in main memory.
        Otherwise, no index data are loaded in main memory and this memory budget is
        applied during queries.
    preload_k_factor_vectors: bool
        When using `k_factor` in a query, we first query for `k_factor * k` pq-encoded vectors,
        and then do a re-ranking step using the original input vectors for the top `k` vectors.
        If `True`, we will load all the input vectors in main memory. This can only be used with
        `memory_budget` set to `-1`, and is useful when the input vectors are small enough to fit in
        memory and you want to speed up re-ranking.
    open_for_remote_query_execution: bool
        If `True`, do not load any index data in main memory locally, and instead load index data in the TileDB Cloud taskgraph created when a non-`None` `driver_mode` is passed to `query()`.
        If `False`, load index data in main memory locally. Note that you can still use a taskgraph for query execution, you'll just end up loading the data both on your local machine and in the cloud taskgraph.
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        memory_budget: int = -1,
        preload_k_factor_vectors: bool = False,
        open_for_remote_query_execution: bool = False,
        group: tiledb.Group = None,
        **kwargs,
    ):
        if preload_k_factor_vectors and memory_budget != -1:
            raise ValueError(
                "preload_k_factor_vectors can only be used with memory_budget set to -1."
            )
        if preload_k_factor_vectors and open_for_remote_query_execution:
            raise ValueError(
                "preload_k_factor_vectors can only be used with open_for_remote_query_execution set to False."
            )

        self.index_open_kwargs = {
            "uri": uri,
            "config": config,
            "timestamp": timestamp,
            "memory_budget": memory_budget,
            "preload_k_factor_vectors": preload_k_factor_vectors,
        }
        self.index_open_kwargs.update(kwargs)
        self.index_type = INDEX_TYPE
        super().__init__(
            uri=uri,
            config=config,
            timestamp=timestamp,
            open_for_remote_query_execution=open_for_remote_query_execution,
            group=group,
        )
        strategy = (
            vspy.IndexLoadStrategy.PQ_INDEX_AND_RERANKING_VECTORS
            if preload_k_factor_vectors
            else vspy.IndexLoadStrategy.PQ_OOC
            if open_for_remote_query_execution
            or (memory_budget != -1 and memory_budget != 0)
            else vspy.IndexLoadStrategy.PQ_INDEX
        )
        self.index = vspy.IndexIVFPQ(
            self.ctx,
            uri,
            strategy,
            0 if memory_budget == -1 else memory_budget,
            to_temporal_policy(timestamp),
        )
        self.db_uri = self.group[
            storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]
        ].uri
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"]
        ].uri
        self.memory_budget = memory_budget

        self.dimensions = self.index.dimensions()
        self.partitions = self.index.partitions()
        self.dtype = np.dtype(self.group.meta.get("dtype", None))
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
        k_factor: float = 1.0,
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
        k_factor: int
            To improve accuracy, IVF_PQ can search for more vectors than requested and then
            perform re-ranking using the original non-PQ-encoded vectors. This can be slightly
            slower, but is more accurate. k_factor is the factor by which to increase the number
            of vectors searched. 1 means we search for exactly `k` vectors. 10 means we search for
            `10*k` vectors.
            Defaults to 1.
        nprobe: int
            Number of partitions to check per query.
            Use this parameter to trade-off accuracy for latency and cost.
        """
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
            queries_feature_vector_array, k=k, nprobe=nprobe, k_factor=k_factor
        )
        return np.array(distances, copy=False), np.array(ids, copy=False)


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    num_subspaces: int,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    distance_metric: vspy.DistanceMetric = vspy.DistanceMetric.SUM_OF_SQUARES,
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
    """
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
    if (
        distance_metric != vspy.DistanceMetric.SUM_OF_SQUARES
        and distance_metric != vspy.DistanceMetric.L2
    ):
        raise ValueError(
            f"Distance metric {distance_metric} is not supported in IVF_PQ"
        )
    vspy.IndexIVFPQ.create(
        ctx=ctx,
        index_uri=uri,
        dimensions=dimensions,
        feature_type=np.dtype(vector_type).name,
        id_type=np.dtype(np.uint64).name,
        partitioning_index_type=np.dtype(np.uint64).name,
        num_subspaces=num_subspaces,
        temporal_policy=vspy.TemporalPolicy(0),
        distance_metric=distance_metric,
        storage_version=storage_version,
    )
    return IVFPQIndex(uri=uri, config=config)
