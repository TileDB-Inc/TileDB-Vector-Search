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
from threading import Thread
from typing import Any, Mapping, Sequence

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
from tiledb.vector_search.utils import create_array_and_add_to_group
from tiledb.vector_search.utils import normalize_vector
from tiledb.vector_search.utils import normalize_vectors

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
        group: tiledb.Group = None,
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
            group=group,
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
        mode: Optional[Mode] = None,
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
            As a rule of thumb, configuring `nprobe` to be the square root of `partitions` should result in accuracy close to 100%.
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

        if self.distance_metric == vspy.DistanceMetric.COSINE:
            queries = normalize_vectors(queries)

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
                    distance_metric=self.distance_metric,
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
                    distance_metric=self.distance_metric,
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
                distance_metric=self.distance_metric,
            )

    def update(self, vector: np.array, external_id: np.uint64, timestamp: int = None):
        if self.distance_metric == vspy.DistanceMetric.COSINE:
            vector = normalize_vector(vector)
        super().update(vector, external_id, timestamp)

    def update_batch(
        self, vectors: np.ndarray, external_ids: np.array, timestamp: int = None
    ):
        if self.distance_metric == vspy.DistanceMetric.COSINE:
            vectors = normalize_vectors(vectors)
        super().update_batch(vectors, external_ids, timestamp)

    def query(
        self,
        queries: np.ndarray,
        k: int,
        **kwargs,
    ):
        if self.distance_metric == vspy.DistanceMetric.COSINE:
            queries = normalize_vectors(queries)
        return super().query(
            queries=queries,
            k=k,
            **kwargs,
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
        distance_metric: vspy.DistanceMetric = vspy.DistanceMetric.SUM_OF_SQUARES,
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
            memory_budget: int = -1,
        ):
            queries_m = array_to_matrix(np.transpose(query_vectors))
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
                upper_bound=0 if memory_budget == -1 else memory_budget,
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
                    memory_budget=self.memory_budget,
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

    def vacuum(self):
        """
        The vacuuming process permanently deletes index files that are consolidated through the consolidation
        process. TileDB separates consolidation from vacuuming, in order to make consolidation process-safe
        in the presence of concurrent reads and writes.

        Note:

        1. Vacuuming is not process-safe and you should take extra care when invoking it.
        2. Vacuuming may affect the granularity of the time traveling functionality.

        The IVFFlat class vacuums consolidated fragment, array metadata and commits for the `db`
        and `ids` arrays.
        """
        super().vacuum()
        if not self.uri.startswith("tiledb://"):
            modes = ["fragment_meta", "commits", "array_meta"]
            for mode in modes:
                conf = tiledb.Config(self.config)
                conf["sm.consolidation.mode"] = mode
                conf["sm.vacuum.mode"] = mode
                tiledb.vacuum(self.db_uri, config=conf)
                tiledb.vacuum(self.ids_uri, config=conf)


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    group_exists: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    distance_metric: vspy.DistanceMetric = vspy.DistanceMetric.SUM_OF_SQUARES,
    group: tiledb.Group = None,
    asset_creation_threads: Sequence[Thread] = None,
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
    group: tiledb.Group
        TileDB group open in write mode.
        Internal, this is used to avoid opening the group multiple times during
        ingestion.
    asset_creation_threads: Sequence[Thread]
        List of asset creation threads to append new threads.
        Internal, this is used to parallelize all asset creation during
        ingestion.

    """
    validate_storage_version(storage_version)
    if (
        distance_metric != vspy.DistanceMetric.SUM_OF_SQUARES
        and distance_metric != vspy.DistanceMetric.L2
        and distance_metric != vspy.DistanceMetric.COSINE
    ):
        raise ValueError(
            f"Distance metric {distance_metric} is not supported in IVF_FLAT"
        )

    if group is None != asset_creation_threads is not None:
        raise ValueError(
            "Can't pass `asset_creation_threads` without a `group` argument."
        )

    with tiledb.scope_ctx(ctx_or_config=config):
        if not group_exists:
            try:
                tiledb.group_create(uri)
            except tiledb.TileDBError as err:
                raise err
        if group is None:
            grp = tiledb.Group(uri, "w")
        else:
            grp = group

        if asset_creation_threads is not None:
            threads = asset_creation_threads
        else:
            threads = []

        index.create_metadata(
            group=grp,
            dimensions=dimensions,
            vector_type=vector_type,
            index_type=INDEX_TYPE,
            storage_version=storage_version,
            distance_metric=distance_metric,
        )

        tile_size = int(TILE_SIZE_BYTES / np.dtype(vector_type).itemsize / dimensions)
        grp.meta["partition_history"] = json.dumps([0])
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
        thread = Thread(
            target=create_array_and_add_to_group,
            kwargs={
                "array_uri": centroids_uri,
                "array_name": centroids_array_name,
                "group": grp,
                "schema": centroids_schema,
            },
        )
        thread.start()
        threads.append(thread)

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
        thread = Thread(
            target=create_array_and_add_to_group,
            kwargs={
                "array_uri": index_array_uri,
                "array_name": index_array_name,
                "group": grp,
                "schema": index_schema,
            },
        )
        thread.start()
        threads.append(thread)

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
        thread = Thread(
            target=create_array_and_add_to_group,
            kwargs={
                "array_uri": ids_uri,
                "array_name": ids_array_name,
                "group": grp,
                "schema": ids_schema,
            },
        )
        thread.start()
        threads.append(thread)

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
        thread = Thread(
            target=create_array_and_add_to_group,
            kwargs={
                "array_uri": parts_uri,
                "array_name": parts_array_name,
                "group": grp,
                "schema": parts_schema,
            },
        )
        thread.start()
        threads.append(thread)

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
        thread = Thread(
            target=create_array_and_add_to_group,
            kwargs={
                "array_uri": updates_array_uri,
                "array_name": updates_array_name,
                "group": grp,
                "schema": updates_schema,
            },
        )
        thread.start()
        threads.append(thread)

        if asset_creation_threads is None:
            for thread in threads:
                thread.join()
        if group is None:
            grp.close()
            return IVFFlatIndex(uri=uri, config=config, memory_budget=1000000)
        else:
            return None
