import json
import multiprocessing
from typing import Any, Mapping

import numpy as np
from tiledb.cloud.dag import Mode

from tiledb.vector_search import index
from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import (STORAGE_VERSION,
                                                  storage_formats)

MAX_INT32 = np.iinfo(np.dtype("int32")).max
TILE_SIZE_BYTES = 64000000  # 64MB
INDEX_TYPE = "IVF_FLAT"


def submit_local(d, func, *args, **kwargs):
    # Drop kwarg
    kwargs.pop("image_name", None)
    kwargs.pop("resources", None)
    return d.submit_local(func, *args, **kwargs)


class IVFFlatIndex(index.Index):
    """
    Open a IVF Flat index

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
        self.index_type = INDEX_TYPE
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

        self.dtype = self.group.meta.get("dtype", None)
        if self.dtype is None:
            schema = tiledb.ArraySchema.load(self.db_uri, ctx=tiledb.Ctx(self.config))
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
        if self.memory_budget == -1:
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

    def query_internal(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int = 1,
        nthreads: int = -1,
        use_nuv_implementation: bool = False,
        mode: Mode = None,
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
            For distributed execution you can use REALTIME or BATCH mode
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

        assert queries.dtype == np.float32

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
            return self.taskgraph_query(
                queries=queries,
                k=k,
                nthreads=nthreads,
                nprobe=nprobe,
                mode=mode,
                num_partitions=num_partitions,
                num_workers=num_workers,
                config=self.config,
            )

    def taskgraph_query(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int = 10,
        nthreads: int = -1,
        mode: Mode = None,
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
            For distributed execution you can use REALTIME or BATCH mode
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
                    resource_class="large" if mode == Mode.REALTIME else None,
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
            results_per_query_d.append(
                np.array(tmp, dtype=np.dtype("float,uint64"))["f0"]
            )
            results_per_query_i.append(
                np.array(tmp, dtype=np.dtype("float,uint64"))["f1"]
            )
        return np.array(results_per_query_d), np.array(results_per_query_i)


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    group_exists: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> IVFFlatIndex:
    index.create_metadata(
        uri=uri,
        dimensions=dimensions,
        vector_type=vector_type,
        index_type=INDEX_TYPE,
        group_exists=group_exists,
        config=config,
    )
    with tiledb.scope_ctx(ctx_or_config=config):
        group = tiledb.Group(uri, "w")
        tile_size = int(TILE_SIZE_BYTES / np.dtype(vector_type).itemsize / dimensions)
        group.meta["partition_history"] = json.dumps([0])
        centroids_array_name = storage_formats[STORAGE_VERSION]["CENTROIDS_ARRAY_NAME"]
        index_array_name = storage_formats[STORAGE_VERSION]["INDEX_ARRAY_NAME"]
        ids_array_name = storage_formats[STORAGE_VERSION]["IDS_ARRAY_NAME"]
        parts_array_name = storage_formats[STORAGE_VERSION]["PARTS_ARRAY_NAME"]
        centroids_uri = f"{uri}/{centroids_array_name}"
        index_array_uri = f"{uri}/{index_array_name}"
        ids_uri = f"{uri}/{ids_array_name}"
        parts_uri = f"{uri}/{parts_array_name}"

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
            filters=storage_formats[STORAGE_VERSION]["DEFAULT_ATTR_FILTERS"],
        )
        centroids_schema = tiledb.ArraySchema(
            domain=centroids_array_dom,
            sparse=False,
            attrs=[centroids_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        tiledb.Array.create(centroids_uri, centroids_schema)
        group.add(centroids_uri, name=centroids_array_name)

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
            filters=storage_formats[STORAGE_VERSION]["DEFAULT_ATTR_FILTERS"],
        )
        index_schema = tiledb.ArraySchema(
            domain=index_array_dom,
            sparse=False,
            attrs=[index_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        tiledb.Array.create(index_array_uri, index_schema)
        group.add(index_array_uri, name=index_array_name)

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
            filters=storage_formats[STORAGE_VERSION]["DEFAULT_ATTR_FILTERS"],
        )
        ids_schema = tiledb.ArraySchema(
            domain=ids_array_dom,
            sparse=False,
            attrs=[ids_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        tiledb.Array.create(ids_uri, ids_schema)
        group.add(ids_uri, name=ids_array_name)

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
            filters=storage_formats[STORAGE_VERSION]["DEFAULT_ATTR_FILTERS"],
        )
        parts_schema = tiledb.ArraySchema(
            domain=parts_array_dom,
            sparse=False,
            attrs=[parts_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        tiledb.Array.create(parts_uri, parts_schema)
        group.add(parts_uri, name=parts_array_name)

        group.close()
        return IVFFlatIndex(uri=uri, config=config, memory_budget=1000000)
