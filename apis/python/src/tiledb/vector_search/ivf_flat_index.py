"""
IVFFlat Index implementation.

IVFFlatIndex is based on `k-means` clustering and shuffling of the dataset vectors.

During ingestion, TileDB computes the `k-means` clusters and shuffles the vectors into partitions.
The vectors are stored grouped by partition in a 2D TileDB array allowing for partitions to be read
with minimal I/O overhead.

To answer a query, the search focuses only on a small number of partitions, based on the query’s proximity
to the `k-means` centroids. This is specified with a parameter called `nprobe` controlling how many partitions
are checked for each query.

IVFFlatIndex provides a vector search implementation that can trade-off accuracy for performance.

Queries can be run in multiple modes:

- Local main memory:
  - Loads the entire index in memory during initialization and uses it to answer queries.
- Local out of core:
  - Avoids loading index data in memory by interleaving I/O and query execution, respecting the
  memory budget defined by the user.
- Distributed execution:
  - Executes the queries using multiple workers in TileDB Cloud.
"""
import json
import multiprocessing
from typing import Any, Mapping

import numpy as np

from tiledb.cloud.dag import Mode
from tiledb.vector_search import index
from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import STORAGE_VERSION
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.storage_formats import validate_storage_version
from tiledb.vector_search.utils import MAX_FLOAT32
from tiledb.vector_search.utils import MAX_INT32
from tiledb.vector_search.utils import MAX_UINT64
from tiledb.vector_search.utils import add_to_group

TILE_SIZE_BYTES = 64000000  # 64MB
INDEX_TYPE = "IVF_FLAT"


def submit_local(d, func, *args, **kwargs):
    # Drop kwarg
    kwargs.pop("image_name", None)
    kwargs.pop("resource_class", None)
    kwargs.pop("resources", None)
    return d.submit_local(func, *args, **kwargs)


class IVFFlatIndex(index.Index):
    """
    Opens an `IVFFlatIndex`.

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
    open_for_remote_query_execution: bool
        If `True`, do not load any index data in main memory locally, and instead load index data in the TileDB Cloud taskgraph created when a non-`None` `driver_mode` is passed to `query()`. We then load index data in the taskgraph based on `memory_budget`.
        If `False`, load index data in main memory locally according to `memory_budget`. Note that you can still use a taskgraph for query execution, you'll just end up loading the data both on your local machine and in the cloud taskgraph..
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        memory_budget: int = -1,
        open_for_remote_query_execution: bool = False,
        **kwargs,
    ):
        self.index_open_kwargs = {
            "uri": uri,
            "config": config,
            "timestamp": timestamp,
            "memory_budget": memory_budget,
        }
        self.index_open_kwargs.update(kwargs)
        self.index_type = INDEX_TYPE
        super().__init__(
            uri=uri,
            config=config,
            timestamp=timestamp,
            open_for_remote_query_execution=open_for_remote_query_execution,
        )
        self.db_uri = self.group[
            storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]
            + self.index_version
        ].uri
        self.centroids_uri = self.group[
            storage_formats[self.storage_version]["CENTROIDS_ARRAY_NAME"]
            + self.index_version
        ].uri
        self.index_array_uri = self.group[
            storage_formats[self.storage_version]["INDEX_ARRAY_NAME"]
            + self.index_version
        ].uri
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"] + self.index_version
        ].uri
        self.memory_budget = memory_budget

        schema = tiledb.ArraySchema.load(self.db_uri, ctx=tiledb.Ctx(self.config))
        self.dimensions = schema.shape[0]

        self.dtype = self.group.meta.get("dtype", None)
        if self.dtype is None:
            self.dtype = np.dtype(schema.attr("values").dtype)
        else:
            self.dtype = np.dtype(self.dtype)

        if self.base_size == 0:
            self.size = 0
            self.partitions = 0
            return

        self.partition_history = [
            int(x)
            for x in list(json.loads(self.group.meta.get("partition_history", "[]")))
        ]
        if len(self.partition_history) == 0:
            schema = tiledb.ArraySchema.load(
                self.centroids_uri, ctx=tiledb.Ctx(self.config)
            )
            self.partitions = schema.domain.dim("cols").domain[1] + 1
        else:
            self.partitions = self.partition_history[self.history_index]

        if not open_for_remote_query_execution:
            self._centroids = load_as_matrix(
                self.centroids_uri,
                ctx=self.ctx,
                size=self.partitions,
                config=config,
                timestamp=self.base_array_timestamp,
            )
            self._index = read_vector_u64(
                self.ctx,
                self.index_array_uri,
                0,
                self.partitions + 1,
                self.base_array_timestamp,
            )

        if self.base_size == -1:
            self.size = self._index[self.partitions]
        else:
            self.size = self.base_size

        # TODO pass in a context
        if not open_for_remote_query_execution and self.memory_budget == -1:
            self._db = load_as_matrix(
                self.db_uri,
                ctx=self.ctx,
                config=config,
                size=self.size,
                timestamp=self.base_array_timestamp,
            )
            self._ids = read_vector_u64(
                self.ctx, self.ids_uri, 0, self.size, self.base_array_timestamp
            )

    def get_dimensions(self):
        """
        Returns the dimension of the vectors in the index.
        """
        return self.dimensions

    def query_internal(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int = 1,
        nthreads: int = -1,
        use_nuv_implementation: bool = False,
        mode: Mode = None,
        resource_class: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        num_partitions: int = -1,
        num_workers: int = -1,
        **kwargs,
    ):
        """
        Queries an `IVFFlatIndex`.

        Parameters
        ----------
        queries: np.ndarray
            2D array of query vectors. This can be used as a batch query interface by passing multiple queries in one call.
        k: int
            Number of results to return per query vector.
        nprobe: int
            Number of partitions to check per query.
            Use this parameter to trade-off accuracy for latency and cost.
        nthreads: int
            Number of threads to use for local query execution.
        use_nuv_implementation: bool
            Whether to use the nuv query implementation. Default: False
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
            return np.full((queries.shape[0], k), MAX_FLOAT32), np.full(
                (queries.shape[0], k), MAX_UINT64
            )

        if mode != Mode.BATCH and resources:
            raise TypeError("Can only pass resources in BATCH mode")
        if (mode != Mode.REALTIME and mode != Mode.BATCH) and resource_class:
            raise TypeError("Can only pass resource_class in REALTIME or BATCH mode")

        if queries.ndim == 1:
            queries = np.array([queries])

        if nthreads == -1:
            nthreads = multiprocessing.cpu_count()

        nprobe = min(nprobe, self.partitions)
        if mode is None:
            queries_m = array_to_matrix(np.transpose(queries))
            if self.memory_budget == -1:
                d, i = ivf_query_ram(
                    self.dtype,
                    self._db,
                    self._centroids,
                    queries_m,
                    self._index,
                    self._ids,
                    nprobe=nprobe,
                    k_nn=k,
                    nthreads=nthreads,
                    ctx=self.ctx,
                    use_nuv_implementation=use_nuv_implementation,
                )
            else:
                d, i = ivf_query(
                    self.dtype,
                    self.db_uri,
                    self._centroids,
                    queries_m,
                    self._index,
                    self.ids_uri,
                    nprobe=nprobe,
                    k_nn=k,
                    memory_budget=self.memory_budget,
                    nthreads=nthreads,
                    ctx=self.ctx,
                    use_nuv_implementation=use_nuv_implementation,
                    timestamp=self.base_array_timestamp,
                )

            return np.transpose(np.array(d)), np.transpose(np.array(i))
        else:
            return self._taskgraph_query(
                queries=queries,
                k=k,
                nthreads=nthreads,
                nprobe=nprobe,
                mode=mode,
                resource_class=resource_class,
                resources=resources,
                num_partitions=num_partitions,
                num_workers=num_workers,
                config=self.config,
            )

    def _taskgraph_query(
        self,
        queries: np.ndarray,
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
        from tiledb.vector_search.module import array_to_matrix
        from tiledb.vector_search.module import dist_qv
        from tiledb.vector_search.module import partition_ivf_index

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
            queries_m = array_to_matrix(np.transpose(query_vectors))
            if timestamp == 0:
                r = dist_qv(
                    dtype=dtype,
                    parts_uri=parts_uri,
                    ids_uri=ids_uri,
                    query_vectors=queries_m,
                    active_partitions=active_partitions,
                    active_queries=active_queries,
                    indices=indices,
                    k_nn=k_nn,
                    ctx=Ctx(config),
                )
            else:
                r = dist_qv(
                    dtype=dtype,
                    parts_uri=parts_uri,
                    ids_uri=ids_uri,
                    query_vectors=queries_m,
                    active_partitions=active_partitions,
                    active_queries=active_queries,
                    indices=indices,
                    k_nn=k_nn,
                    ctx=Ctx(config),
                    timestamp=timestamp,
                )
            results = []
            for q in range(len(r)):
                tmp_results = []
                for j in range(len(r[q])):
                    tmp_results.append(r[q][j])
                results.append(tmp_results)
            return results

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
                    resource_class="large"
                    if (not resources and not resource_class)
                    else resource_class,
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


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    group_exists: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    **kwargs,
) -> IVFFlatIndex:
    """
    Creates an empty IVFFlatIndex.

    Parameters
    ----------
    uri: str
        URI of the index.
    dimensions: int
        Number of dimensions for the vectors to be stored in the index.
    vector_type: np.dtype
        Datatype of vectors.
        Supported values (uint8, int8, float32).
    group_exists: bool
        If False it creates the TileDB group for the index.
        If True the method expects the TileDB group to be already created.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    storage_version: str
        The TileDB vector search storage version to use.
        If not provided, use the latest stable storage version.
    """
    validate_storage_version(storage_version)

    index.create_metadata(
        uri=uri,
        dimensions=dimensions,
        vector_type=vector_type,
        index_type=INDEX_TYPE,
        storage_version=storage_version,
        group_exists=group_exists,
        config=config,
    )
    with tiledb.scope_ctx(ctx_or_config=config):
        group = tiledb.Group(uri, "w")
        tile_size = int(TILE_SIZE_BYTES / np.dtype(vector_type).itemsize / dimensions)
        group.meta["partition_history"] = json.dumps([0])
        centroids_array_name = storage_formats[storage_version]["CENTROIDS_ARRAY_NAME"]
        index_array_name = storage_formats[storage_version]["INDEX_ARRAY_NAME"]
        ids_array_name = storage_formats[storage_version]["IDS_ARRAY_NAME"]
        parts_array_name = storage_formats[storage_version]["PARTS_ARRAY_NAME"]
        updates_array_name = storage_formats[storage_version]["UPDATES_ARRAY_NAME"]
        centroids_uri = f"{uri}/{centroids_array_name}"
        index_array_uri = f"{uri}/{index_array_name}"
        ids_uri = f"{uri}/{ids_array_name}"
        parts_uri = f"{uri}/{parts_array_name}"
        updates_array_uri = f"{uri}/{updates_array_name}"

        centroids_array_rows_dim = tiledb.Dim(
            name="rows",
            domain=(0, dimensions - 1),
            tile=dimensions,
            dtype=np.dtype(np.int32),
        )
        centroids_array_cols_dim = tiledb.Dim(
            name="cols",
            domain=(0, MAX_INT32),
            tile=100000,
            dtype=np.dtype(np.int32),
        )
        centroids_array_dom = tiledb.Domain(
            centroids_array_rows_dim, centroids_array_cols_dim
        )
        centroids_attr = tiledb.Attr(
            name="centroids",
            dtype=np.dtype(np.float32),
            filters=storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"],
        )
        centroids_schema = tiledb.ArraySchema(
            domain=centroids_array_dom,
            sparse=False,
            attrs=[centroids_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        tiledb.Array.create(centroids_uri, centroids_schema)
        add_to_group(group, centroids_uri, name=centroids_array_name)

        index_array_rows_dim = tiledb.Dim(
            name="rows",
            domain=(0, MAX_INT32),
            tile=100000,
            dtype=np.dtype(np.int32),
        )
        index_array_dom = tiledb.Domain(index_array_rows_dim)
        index_attr = tiledb.Attr(
            name="values",
            dtype=np.dtype(np.uint64),
            filters=storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"],
        )
        index_schema = tiledb.ArraySchema(
            domain=index_array_dom,
            sparse=False,
            attrs=[index_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        tiledb.Array.create(index_array_uri, index_schema)
        add_to_group(group, index_array_uri, name=index_array_name)

        ids_array_rows_dim = tiledb.Dim(
            name="rows",
            domain=(0, MAX_INT32),
            tile=tile_size,
            dtype=np.dtype(np.int32),
        )
        ids_array_dom = tiledb.Domain(ids_array_rows_dim)
        ids_attr = tiledb.Attr(
            name="values",
            dtype=np.dtype(np.uint64),
            filters=storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"],
        )
        ids_schema = tiledb.ArraySchema(
            domain=ids_array_dom,
            sparse=False,
            attrs=[ids_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        tiledb.Array.create(ids_uri, ids_schema)
        add_to_group(group, ids_uri, name=ids_array_name)

        parts_array_rows_dim = tiledb.Dim(
            name="rows",
            domain=(0, dimensions - 1),
            tile=dimensions,
            dtype=np.dtype(np.int32),
        )
        parts_array_cols_dim = tiledb.Dim(
            name="cols",
            domain=(0, MAX_INT32),
            tile=tile_size,
            dtype=np.dtype(np.int32),
        )
        parts_array_dom = tiledb.Domain(parts_array_rows_dim, parts_array_cols_dim)
        parts_attr = tiledb.Attr(
            name="values",
            dtype=vector_type,
            filters=storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"],
        )
        parts_schema = tiledb.ArraySchema(
            domain=parts_array_dom,
            sparse=False,
            attrs=[parts_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        tiledb.Array.create(parts_uri, parts_schema)
        add_to_group(group, parts_uri, name=parts_array_name)

        external_id_dim = tiledb.Dim(
            name="external_id",
            domain=(0, MAX_UINT64 - 1),
            dtype=np.dtype(np.uint64),
        )
        dom = tiledb.Domain(external_id_dim)
        vector_attr = tiledb.Attr(name="vector", dtype=vector_type, var=True)
        updates_schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[vector_attr],
            allows_duplicates=False,
        )
        tiledb.Array.create(updates_array_uri, updates_schema)
        add_to_group(group, updates_array_uri, name=updates_array_name)

        group.close()
        return IVFFlatIndex(uri=uri, config=config, memory_budget=1000000)
