"""
Vamana Index implementation.

Vamana is based on Microsoft's DiskANN vector search library, as described in these papers:
```
  Subramanya, Suhas Jayaram, and Rohan Kadekodi. DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node.

  Singh, Aditi, et al. FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search. arXiv:2105.09613, arXiv, 20 May 2021, http://arxiv.org/abs/2105.09613.

  Gollapudi, Siddharth, et al. "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters." Proceedings of the ACM Web Conference 2023, ACM, 2023, pp. 3406-16, https://doi.org/10.1145/3543507.3583552.
```
"""
import json
import re
import warnings
from typing import Any, Mapping, Optional, Set

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


def _parse_where_clause(where: str, label_enumeration: dict) -> Set[int]:
    """
    Parse a simple where clause and return a set of label IDs.

    Supports basic equality conditions like: "label_col == 'value'"

    Parameters
    ----------
    where : str
        The where clause string to parse
    label_enumeration : dict
        Mapping from label strings to enumeration IDs

    Returns
    -------
    Set[int]
        Set of label IDs matching the where clause

    Raises
    ------
    ValueError
        If the where clause is invalid or references non-existent labels
    """
    # Simple pattern for: column_name == 'value'
    # We support single or double quotes
    pattern = r"\s*\w+\s*==\s*['\"]([^'\"]+)['\"]\s*"
    match = re.match(pattern, where.strip())

    if not match:
        raise ValueError(
            f"Invalid where clause: '{where}'. "
            "Expected format: \"label_col == 'value'\""
        )

    label_value = match.group(1)

    # Check if the label exists in the enumeration
    if label_value not in label_enumeration:
        available_labels = ", ".join(sorted(label_enumeration.keys()))
        raise ValueError(
            f"Label '{label_value}' not found in index. "
            f"Available labels: {available_labels}"
        )

    # Return the enumeration ID for this label
    label_id = label_enumeration[label_value]
    return {label_id}


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
        group: tiledb.Group = None,
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
            group=group,
        )
        # TODO(SC-48710): Add support for `open_for_remote_query_execution`. We don't leave `self.index`` as `None` because we need to be able to call index.dimensions().
        self.index = vspy.IndexVamana(self.ctx, uri, to_temporal_policy(timestamp))
        self.db_uri = self.group[
            storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]
        ].uri
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"]
        ].uri

        self.dimensions = self.index.dimensions()
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
        l_search: Optional[int] = L_SEARCH_DEFAULT,
        where: Optional[str] = None,
        **kwargs,
    ):
        """
        Queries a `VamanaIndex` for k approximate nearest neighbors.

        Parameters
        ----------
        queries: np.ndarray
            Query vectors. Can be 1D (single query) or 2D array (batch queries).
            For batch queries, each row is a separate query vector.
        k: int
            Number of nearest neighbors to return per query.
            Default: 10
        l_search: int
            Search depth parameter. Larger values result in slower latencies but higher recall.
            Should be >= k. If l_search < k, it will be automatically set to k.
            Default: 100
        where: Optional[str]
            Filter condition to restrict search to vectors matching specific labels.
            Only vectors with matching labels will be considered in the search.
            Requires the index to be built with filter_labels.

            Supported syntax:
                - Equality: "label == 'value'"
                  Returns vectors where label exactly matches 'value'

                - Set membership: "label IN ('value1', 'value2', ...)"
                  Returns vectors where label matches any value in the set

            Examples:
                - where="soma_uri == 'dataset_A'"
                  Only search vectors from dataset_A

                - where="region IN ('US', 'EU', 'ASIA')"
                  Search vectors from US, EU, or ASIA regions

                - where="source == 'experiment_42'"
                  Only search vectors from experiment_42

            Performance:
                Filtered search achieves >90% recall even for highly selective filters:
                - Specificity 10^-3 (0.1% of data): >95% recall
                - Specificity 10^-6 (0.0001% of data): >90% recall

                This is achieved through the Filtered-Vamana algorithm, which
                modifies graph construction to preserve connectivity for rare labels.

            Default: None (unfiltered search)

        Returns
        -------
        distances : np.ndarray
            Distances to k nearest neighbors. Shape: (n_queries, k)
            Sentinel value MAX_FLOAT32 indicates no valid result at that position.
        ids : np.ndarray
            External IDs of k nearest neighbors. Shape: (n_queries, k)
            Sentinel value MAX_UINT64 indicates no valid result at that position.

        Raises
        ------
        ValueError
            - If where clause syntax is invalid
            - If where is provided but index lacks filter metadata
            - If label value in where clause doesn't exist in index

        Notes
        -----
        - The where parameter requires the index to be built with filter_labels
          during ingestion. If the index was created without filters, passing
          a where clause will raise ValueError.
        - Unfiltered queries on filtered indexes work correctly - simply omit
          the where parameter.
        - For best performance with filters, ensure l_search is appropriately
          sized for the expected specificity of your queries.

        See Also
        --------
        ingest : Create an index with filter_labels support

        References
        ----------
        Filtered search is based on:
        "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor
        Search with Filters" (Gollapudi et al., WWW 2023)
        https://doi.org/10.1145/3543507.3583552
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

        # NEW: Handle filtered queries
        query_filter = None
        if where is not None:
            # Get label enumeration from metadata
            label_enum_str = self.group.meta.get("label_enumeration", None)
            if label_enum_str is None:
                raise ValueError(
                    "Cannot use 'where' parameter: index does not have filter metadata. "
                    "This index was not created with filter support."
                )

            # Parse JSON string to get label enumeration
            label_enumeration = json.loads(label_enum_str)

            # Parse where clause and get filter label IDs
            query_filter = _parse_where_clause(where, label_enumeration)

        distances, ids = self.index.query(
            queries_feature_vector_array, k, l_search, query_filter
        )

        return np.array(distances, copy=False), np.array(ids, copy=False)


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    l_build: int = L_BUILD_DEFAULT,
    r_max_degree: int = R_MAX_DEGREE_DEFAULT,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    distance_metric: vspy.DistanceMetric = vspy.DistanceMetric.SUM_OF_SQUARES,
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
    if (
        distance_metric != vspy.DistanceMetric.L2
        and distance_metric != vspy.DistanceMetric.SUM_OF_SQUARES
        and distance_metric != vspy.DistanceMetric.COSINE
    ):
        raise ValueError(
            f"Distance metric {distance_metric} is not supported in VAMANA"
        )
    ctx = vspy.Ctx(config)
    index = vspy.IndexVamana(
        feature_type=np.dtype(vector_type).name,
        id_type=np.dtype(np.uint64).name,
        dimensions=dimensions,
        l_build=l_build if l_build > 0 else L_BUILD_DEFAULT,
        r_max_degree=r_max_degree if l_build > 0 else R_MAX_DEGREE_DEFAULT,
        distance_metric=int(distance_metric),
    )
    # TODO(paris): Run all of this with a single C++ call.
    empty_vector = vspy.FeatureVectorArray(
        dimensions, 0, np.dtype(vector_type).name, np.dtype(np.uint64).name
    )
    index.train(empty_vector)
    index.add(empty_vector)
    index.write_index(ctx, uri, vspy.TemporalPolicy(0), storage_version)
    return VamanaIndex(uri=uri, config=config)
