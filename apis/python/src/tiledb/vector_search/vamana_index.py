import json
import multiprocessing
from typing import Any, Mapping

import numpy as np
from tiledb.cloud.dag import Mode

from tiledb.vector_search import index
from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import (STORAGE_VERSION,
                                                  storage_formats,
                                                  validate_storage_version)
from tiledb.vector_search.utils import add_to_group
from tiledb.vector_search import _tiledbvspy as vspy

MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
INDEX_TYPE = "VAMANA"

class VamanaIndex(index.Index):
    """
    Open a Vamana index

    Parameters
    ----------
    uri: str
        URI of the index
    config: Optional[Mapping[str, Any]]
        config dictionary, defaults to None
    memory_budget: int
        Main memory budget. If not provided, no memory budget is applied.
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        **kwargs,
    ):
        super().__init__(uri=uri, config=config, timestamp=timestamp)
        self.index_type = INDEX_TYPE
        self.index = vspy.IndexVamana(vspy.Ctx(config), uri)
        self.db_uri = self.group[
            storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]
            + self.index_version
        ].uri
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"] 
            + self.index_version
        ].uri
        
        schema = tiledb.ArraySchema.load(self.db_uri, ctx=tiledb.Ctx(self.config))
        self.dimensions = self.index.dimension()
        
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
        return self.dimensions

    def query_internal(
        self,
        queries: np.ndarray,
        k: int = 10,
        nthreads: int = -1,
        mode: Mode = None,
        resource_class: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None
    ):
        """
        Query an VAMANA index

        Parameters
        ----------
        queries: numpy.ndarray
            ND Array of queries
        k: int
            Number of top results to return per query
        nthreads: int
            Number of threads to use for query
        mode: Mode
            If provided the query will be executed using TileDB cloud taskgraphs.
            For distributed execution you can use REALTIME or BATCH mode. 
            For local execution you can use LOCAL mode.
        resource_class: 
            The name of the resource class to use ("standard" or "large"). Resource classes define maximum
            limits for cpu and memory usage. Can only be used in REALTIME or BATCH mode.
            Cannot be used alongside resources.
            In REALTIME or BATCH mode if neither resource_class nor resources are provided,
            we default to the "large" resource class.
        resources:
            A specification for the amount of resources to use when executing using TileDB cloud 
            taskgraphs, of the form: {"cpu": "6", "memory": "12Gi", "gpu": 1}. Can only be used 
            in BATCH mode. Cannot be used alongside resource_class.
        """
        if self.size == 0:
            return np.full((queries.shape[0], k), index.MAX_FLOAT_32), np.full(
                (queries.shape[0], k), index.MAX_UINT64
            )

        if mode != Mode.BATCH and resources:
            raise TypeError("Can only pass resources in BATCH mode")
        if (mode != Mode.REALTIME and mode != Mode.BATCH) and resource_class:
            raise TypeError("Can only pass resource_class in REALTIME or BATCH mode")

        assert queries.dtype == np.float32

        if queries.ndim == 1:
            queries = np.array([queries])

        if nthreads == -1:
            nthreads = multiprocessing.cpu_count()

        # TODO(paris): Actually run the query.
        return [], []

# TODO(paris): Pass more arguments to C++, i.e. storage_version.
def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    id_type: np.dtype = np.uint32,
    adjacency_row_index_type: np.dtype = np.uint32,
    group_exists: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    **kwargs,
) -> VamanaIndex:
      if not group_exists:
        ctx = vspy.Ctx(config)
        index = vspy.IndexVamana(
            feature_type=np.dtype(vector_type).name, 
            id_type=np.dtype(id_type).name, 
            adjacency_row_index_type=np.dtype(adjacency_row_index_type).name, 
            dimension=dimensions,
        )
        empty_vector = vspy.FeatureVectorArray(
            dimensions, 
            0, 
            np.dtype(vector_type).name, 
            np.dtype(id_type).name
            )
        index.train(empty_vector)
        index.add(empty_vector)
        index.write_index(ctx, uri)
      return VamanaIndex(uri=uri, config=config, memory_budget=1000000)