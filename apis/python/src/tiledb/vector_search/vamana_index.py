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
        debug=False,
        **kwargs,
    ):
        print('[vamana_index@__init__] uri', uri)
        super().__init__(uri=uri, config=config, timestamp=timestamp)
        self.index_type = INDEX_TYPE
        # if debug:
        #     return
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
        
        print('[vamana_index@__init__] self.base_size', self.base_size)
        print('[vamana_index@__init__] schema.domain', schema.domain)
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
        opt_l: Optional[int] = 1,
        nthreads: int = 8,
    ):
        """
        Query an VAMANA index

        Parameters
        ----------
        queries: numpy.ndarray
            ND Array of queries
        k: int
            Number of top results to return per query
        opt_l: int
            How deep to search
        nthreads: int
            Number of threads to use for query
        """
        print('[vamana_index@query_internal] self.size', self.size)
        if self.size == 0:
            return np.full((queries.shape[0], k), index.MAX_FLOAT_32), np.full(
                (queries.shape[0], k), index.MAX_UINT64
            )

        assert queries.dtype == np.float32

        if queries.ndim == 1:
            queries = np.array([queries])

        # queries_m = array_to_matrix(np.transpose(queries))
        print('[vamana_index@query_internal] queries', queries)
        queries_feature_vector_array = vspy.FeatureVectorArray(np.transpose(queries))
        print('[vamana_index@query_internal] queries_feature_vector_array.dimension(): ', queries_feature_vector_array.dimension())
        print('[vamana_index@query_internal] queries_feature_vector_array.num_vectors(): ', queries_feature_vector_array.num_vectors())
        print('[vamana_index@query_internal] queries_feature_vector_array.feature_type_string(): ', queries_feature_vector_array.feature_type_string())
        scores, ids = self.index.query(queries_feature_vector_array, k, opt_l)
        print('[vamana_index@query_internal] scores', scores)
        print('[vamana_index@query_internal] ids', ids)
        
        print('[vamana_index@query_internal] scores.dimension(): ', scores.dimension())
        print('[vamana_index@query_internal] scores.num_vectors(): ', scores.num_vectors())
        print('[vamana_index@query_internal] scores.feature_type_string(): ', scores.feature_type_string())

        print('[vamana_index@query_internal] ids.dimension(): ', ids.dimension())
        print('[vamana_index@query_internal] ids.num_vectors(): ', ids.num_vectors())
        print('[vamana_index@query_internal] ids.feature_type_string(): ', ids.feature_type_string())
        
        # return np.transpose(np.array(d)), np.transpose(np.array(i))
        return np.array(scores, copy=False), np.array(ids, copy=False)

# TODO(paris): Pass more arguments to C++, i.e. storage_version.
def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    id_type: np.dtype = np.uint32,
    adjacency_row_index_type: np.dtype = np.uint32,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    **kwargs,
) -> VamanaIndex:
      ctx = vspy.Ctx(config)
      index = vspy.IndexVamana(
          feature_type=np.dtype(vector_type).name, 
          id_type=np.dtype(id_type).name, 
          adjacency_row_index_type=np.dtype(adjacency_row_index_type).name, 
          dimension=dimensions,
      )
      # TODO(paris): Run all of this with a single C++ call.
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