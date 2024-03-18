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

MAX_INT32 = np.iinfo(np.dtype("int32")).max
MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
TILE_SIZE_BYTES = 64000000  # 64MB
INDEX_TYPE = "VAMANA"


def submit_local(d, func, *args, **kwargs):
    # Drop kwarg
    kwargs.pop("image_name", None)
    kwargs.pop("resource_class", None)
    kwargs.pop("resources", None)
    return d.submit_local(func, *args, **kwargs)


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
        print('[vamana_index@__init__]')
        super().__init__(uri=uri, config=config, timestamp=timestamp)
        self.index_type = INDEX_TYPE
        self.index = vspy.IndexVamana(vspy.Ctx(config), uri)
        self.db_uri = self.group[
            storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]
            + self.index_version
        ].uri
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"] + self.index_version
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
        
        print('[vamana_index@__init__] self.size', self.size)
        print('[vamana_index@__init__] self.dtype', self.dtype)
        print('[vamana_index@__init__] self.dimensions', self.dimensions)
        print('[vamana_index@__init__] self.db_uri', self.db_uri)
        print('[vamana_index@__init__] self.ids_uri', self.ids_uri)

        print('[vamana_index@__init__] done')

    def get_dimensions(self):
        return self.dimensions

    def query_internal(
        self,
        queries: np.ndarray,
        k: int = 10,
        opt_l: Optional[int] = None,
        nprobe: int = 1,
        nthreads: int = -1,
        use_nuv_implementation: bool = False,
        mode: Mode = None,
        resource_class: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        num_partitions: int = -1,
        num_workers: int = -1,
    ):
        """
        Query an VAMANA index

        Parameters
        ----------
        queries: numpy.ndarray
            ND Array of queries
        k: int
            Number of top results to return per query
        nprobe: int
            number of probes
        nthreads: int
            Number of threads to use for query
        use_nuv_implementation: bool
            wether to use the nuv query implementation. Default: False
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
        num_partitions: int
            Only relevant for taskgraph based execution.
            If provided, we split the query execution in that many partitions.
        num_workers: int
            Only relevant for taskgraph based execution.
            If provided, this is the number of workers to use for the query execution.

        """
        print('[vamana_index@query_internal]')
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

        nprobe = min(nprobe, self.partitions)
        if mode is None:
            # queries_m = array_to_matrix(np.transpose(queries))
            queries = vspy.FeatureVectorArray(vspy.Ctx(self.config), queries)
            scores, ids = self.index.query(queries, k, opt_l)
            print('[vamana_index.py] scores', scores, 'ids', ids)
            # return np.transpose(np.array(d)), np.transpose(np.array(i))
            return scores, ids
        else:
            # TODO(paris): Support taskgraph queries.
            return [], []

# TODO(paris): Pass more arugments to C++, i.e. storage_version.
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
      print('[vamana_index@create]')
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