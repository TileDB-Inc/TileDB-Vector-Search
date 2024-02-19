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
        memory_budget: int = -1,
        **kwargs,
    ):
        super().__init__(uri=uri, config=config, timestamp=timestamp)
        print('[vamana_index] uri', uri)
        self.index = vspy.IndexVamana(tiledb.Ctx(config), uri)
        self.index_type = INDEX_TYPE

    def get_dimensions(self):
        # return self.dimensions
        raise NotImplementedError

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
        Query an IVF_FLAT index

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
            queries = vspy.FeatureVectorArray(Ctx(self.config), queries)
            d, i = self.index.query(queries, k, opt_l)
            # return np.transpose(np.array(d)), np.transpose(np.array(i))
            return d, i
        else:
            return [], []
            # return self.taskgraph_query(
            #     queries=queries,
            #     k=k,
            #     opt_l=opt_l,
            #     nthreads=nthreads,
            #     nprobe=nprobe,
            #     mode=mode,
            #     resource_class=resource_class,
            #     resources=resources,
            #     num_partitions=num_partitions,
            #     num_workers=num_workers,
            #     config=self.config,
            # )

    def taskgraph_query(
        self,
        queries: np.ndarray,
        opt_l: Optional[int],
        k: int = 10,
        nprobe: int = 10,
        nthreads: int = -1,
        mode: Mode = None,
        resource_class: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        num_partitions: int = -1,
        num_workers: int = -1,
        config: Optional[Mapping[str, Any]] = None,
    ):
        """
        Query an IVF_FLAT index using TileDB cloud taskgraphs

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
        config: None
            config dictionary, defaults to None
        """
        import math
        from functools import partial

        import numpy as np
        from tiledb.cloud import dag
        from tiledb.cloud.dag import Mode

        from tiledb.vector_search.module import (array_to_matrix, dist_qv,
                                                 partition_ivf_index)

        if resource_class and resources:
            raise TypeError("Cannot provide both resource_class and resources")

        def dist_qv_udf(
            dtype: np.dtype,
            parts_uri: str,
            ids_uri: str,
            query_vectors: np.ndarray,
            active_partitions: np.array,
            active_queries: np.array,
            indices: np.array,
            k_nn: int,
            config: Optional[Mapping[str, Any]] = None,
            timestamp: int = 0,
        ):
            queries = vspy.FeatureVectorArray(Ctx(self.config), queries)
            return self.index.query(queries, k, opt_l)


        assert queries.dtype == np.float32
        if num_partitions == -1:
            num_partitions = 5
        if num_workers == -1:
            num_workers = num_partitions
        if mode == Mode.BATCH:
            d = dag.DAG(
                name="vector-query",
                mode=Mode.BATCH,
                max_workers=num_workers,
            )
        elif mode == Mode.REALTIME:
            d = dag.DAG(
                name="vector-query",
                mode=Mode.REALTIME,
                max_workers=num_workers,
            )
        else:
            d = dag.DAG(
                name="vector-query",
                mode=Mode.REALTIME,
                max_workers=1,
                namespace="default",
            )
        submit = partial(submit_local, d)
        if mode == Mode.BATCH or mode == Mode.REALTIME:
            submit = d.submit

        queries_m = array_to_matrix(np.transpose(queries))
        active_partitions, active_queries = partition_ivf_index(
            centroids=self._centroids, query=queries_m, nprobe=nprobe, nthreads=nthreads
        )
        num_parts = len(active_partitions)

        parts_per_node = int(math.ceil(num_parts / num_partitions))
        nodes = []
        for part in range(0, num_parts, parts_per_node):
            part_end = part + parts_per_node
            if part_end > num_parts:
                part_end = num_parts
            aq = []
            for tt in range(part, part_end):
                aqt = []
                for ttt in range(len(active_queries[tt])):
                    aqt.append(active_queries[tt][ttt])
                aq.append(aqt)
            nodes.append(
                submit(
                    dist_qv_udf,
                    dtype=self.dtype,
                    parts_uri=self.db_uri,
                    ids_uri=self.ids_uri,
                    query_vectors=queries,
                    active_partitions=np.array(active_partitions)[part:part_end],
                    active_queries=np.array(aq, dtype=object),
                    indices=np.array(self._index),
                    k_nn=k,
                    config=config,
                    timestamp=self.base_array_timestamp,
                    resource_class="large" if (not resources and not resource_class) else resource_class,
                    resources=resources,
                    image_name="3.9-vectorsearch",
                )
            )

        d.compute()
        d.wait()
        results = []
        for node in nodes:
            res = node.result()
            results.append(res)

        results_per_query_d = []
        results_per_query_i = []
        for q in range(queries.shape[0]):
            tmp_results = []
            for j in range(k):
                for r in results:
                    if len(r[q]) > j:
                        if r[q][j][0] > 0:
                            tmp_results.append(r[q][j])
            tmp = sorted(tmp_results, key=lambda t: t[0])[0:k]
            for j in range(len(tmp), k):
                tmp.append((float(0.0), int(0)))
            results_per_query_d.append(np.array(tmp, dtype=np.float32)[:, 0])
            results_per_query_i.append(np.array(tmp, dtype=np.uint64)[:, 1])
        return np.array(results_per_query_d), np.array(results_per_query_i)

# TODO(paris): Look into passing args to C++, i.e. storage_version.
def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    id_type: np.dtype = np.uint32,
    px_type: np.dtype = np.uint32,
    group_exists: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    **kwargs,
) -> VamanaIndex:
      ctx = vspy.Ctx(config)
      index = vspy.IndexVamana(
          feature_type=np.dtype(vector_type).name, 
          id_type=np.dtype(id_type).name, 
          px_type=np.dtype(px_type).name, 
          dimension=dimensions,
        )
      index.write_index(ctx, uri)
      return VamanaIndex(uri=uri, config=config, memory_budget=1000000)