import multiprocessing
import os
import math
import logging

import numpy as np
from tiledb.vector_search.module import *
from tiledb.cloud.dag import Mode
from typing import Any, Mapping

CENTROIDS_ARRAY_NAME = "centroids.tdb"
INDEX_ARRAY_NAME = "index.tdb"
IDS_ARRAY_NAME = "ids.tdb"
PARTS_ARRAY_NAME = "parts.tdb"


def submit_local(d, func, *args, **kwargs):
    # Drop kwarg
    kwargs.pop("image_name", None)
    kwargs.pop("resources", None)
    return d.submit_local(func, *args, **kwargs)


class Index:
    def query(self, targets: np.ndarray, k=10, nqueries=10, nthreads=8, nprobe=1):
        raise NotImplementedError


class FlatIndex(Index):
    """
    Open a flat index

    Parameters
    ----------
    uri: str
        URI of datataset
    dtype: numpy.dtype
        datatype float32 or uint8
    parts_name: str
        Optional name of partitions
    """

    def __init__(
        self,
        uri: str,
        dtype: Optional[np.dtype] = None,
        parts_name: str = "parts.tdb",
        config: Optional[Mapping[str, Any]] = None,
    ):
        # If the user passes a tiledb python Config object convert to a dictionary
        if isinstance(config, tiledb.Config):
            config = dict(config)

        self.uri = uri
        self.dtype = dtype
        self._index = None
        self.ctx = Ctx(config)
        self.config = config

        self._db = load_as_matrix(os.path.join(uri, parts_name), ctx=self.ctx, config=config)

        if dtype is None:
            self.dtype = self._db.dtype
        else:
            self.dtype = dtype

    def query(
        self,
        targets: np.ndarray,
        k: int = 10,
        nthreads: int = 8,
        nprobe: int = 1,
        query_type="heap",
    ):
        """
        Query a flat index

        Parameters
        ----------
        targets: numpy.ndarray
            ND Array of query targets
        k: int
            Number of top results to return per target
        nqueries: int
            Number of queries
        nthreads: int
            Number of threads to use for queyr
        nprobe: int
            number of probes
        """
        # TODO:
        # - typecheck targets
        # - add all the options and query strategies

        assert targets.dtype == np.float32

        targets_m = array_to_matrix(np.transpose(targets))

        if query_type == "heap":
            r = query_vq_heap(self._db, targets_m, k, nthreads)
        elif query_type == "nth":
            r = query_vq_nth(self._db, targets_m, k, nthreads)
        else:
            raise Exception("Unknown query type!")

        return np.transpose(np.array(r))


class IVFFlatIndex(Index):
    """
    Open a IVF Flat index

    Parameters
    ----------
    uri: str
        URI of datataset
    dtype: numpy.dtype
        datatype float32 or uint8
    memory_budget: int
        Main memory budget. If not provided no memory budget is applied.
    """

    def __init__(
        self,
        uri,
        dtype: np.dtype = None,
        memory_budget: int = -1,
        config: Optional[Mapping[str, Any]] = None,
    ):
        # If the user passes a tiledb python Config object convert to a dictionary
        if isinstance(config, tiledb.Config):
            config = dict(config)

        self.config = config
        self.ctx = Ctx(config)
        group = tiledb.Group(uri, ctx=tiledb.Ctx(config))
        self.parts_db_uri = group[PARTS_ARRAY_NAME].uri
        self.centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
        self.index_uri = group[INDEX_ARRAY_NAME].uri
        self.ids_uri = group[IDS_ARRAY_NAME].uri
        self.memory_budget = memory_budget

        # TODO pass in a context
        if self.memory_budget == -1:
            self._db = load_as_matrix(self.parts_db_uri, ctx=self.ctx, config=config)
            self._ids = read_vector_u64(self.ctx, self.ids_uri)

        self._centroids = load_as_matrix(self.centroids_uri, ctx=self.ctx, config=config)

        # TODO this should always be available
        if dtype is None:
            self.dtype = self._centroids.dtype
        else:
            self.dtype = dtype
        self._index = read_vector_u64(self.ctx, self.index_uri)

    def query(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int = 10,
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
            Number of top results to return per target
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
        assert queries.dtype == np.float32

        if queries.ndim == 1:
            queries = np.array([queries])

        if nthreads == -1:
            nthreads = multiprocessing.cpu_count()
        if mode is None:
            queries_m = array_to_matrix(np.transpose(queries))
            if self.memory_budget == -1:
                r = ivf_query_ram(
                    self.dtype,
                    self._db,
                    self._centroids,
                    queries_m,
                    self._index,
                    self._ids,
                    nprobe=nprobe,
                    k_nn=k,
                    nth=True,  # ??
                    nthreads=nthreads,
                    ctx=self.ctx,
                    use_nuv_implementation=use_nuv_implementation,
                )
            else:
                r = ivf_query(
                    self.dtype,
                    self.parts_db_uri,
                    self._centroids,
                    queries_m,
                    self._index,
                    self.ids_uri,
                    nprobe=nprobe,
                    k_nn=k,
                    memory_budget=self.memory_budget,
                    nth=True,  # ??
                    nthreads=nthreads,
                    ctx=self.ctx,
                    use_nuv_implementation=use_nuv_implementation,
                )

            return np.transpose(np.array(r))
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
            Number of top results to return per target
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
        from tiledb.cloud import dag
        from tiledb.cloud.dag import Mode
        from tiledb.vector_search.module import (
            array_to_matrix,
            partition_ivf_index,
            dist_qv,
        )
        import math
        import numpy as np
        from functools import partial

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
                ctx=Ctx(config)
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
        if mode == Mode.REALTIME:
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
                    parts_uri=self.parts_db_uri,
                    ids_uri=self.ids_uri,
                    query_vectors=queries,
                    active_partitions=np.array(active_partitions)[part:part_end],
                    active_queries=np.array(
                        aq, dtype=object
                    ),
                    indices=np.array(self._index),
                    k_nn=k,
                    config=config,
                    resource_class="large",
                    image_name="3.9-vectorsearch",
                )
            )

        d.compute()
        d.wait()
        results = []
        for node in nodes:
            res = node.result()
            results.append(res)

        results_per_query = []
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
            results_per_query.append(np.array(tmp, dtype=np.dtype('float,int'))['f1'])
        return results_per_query
