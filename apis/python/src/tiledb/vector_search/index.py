import os
import math

import numpy as np
from tiledb.vector_search.module import *
from tiledb.cloud.dag import Mode

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

    def __init__(self, uri: str, dtype: np.dtype, parts_name: str = "parts.tdb"):
        self.uri = uri
        self.dtype = dtype
        self._index = None

        self._db = load_as_matrix(os.path.join(uri, parts_name))

    def query(
        self,
        targets: np.ndarray,
        k: int = 10,
        nqueries: int = 10,
        nthreads: int = 8,
        nprobe: int = 1,
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

        targets_m = array_to_matrix(targets)

        r = query_vq(self._db, targets_m, k, nqueries, nthreads)
        return np.array(r)


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
        self, uri, dtype: np.dtype, memory_budget: int = -1, ctx: "Ctx" = None
    ):
        group = tiledb.Group(uri)
        self.parts_db_uri = group[PARTS_ARRAY_NAME].uri
        self.centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
        self.index_uri = group[INDEX_ARRAY_NAME].uri
        self.ids_uri = group[IDS_ARRAY_NAME].uri
        self.dtype = dtype
        self.memory_budget = memory_budget
        self.ctx = ctx
        if ctx is None:
            self.ctx = Ctx({})

        # TODO pass in a context
        if self.memory_budget == -1:
            self._db = load_as_matrix(self.parts_db_uri)
            self._ids = read_vector_u64(self.ctx, self.ids_uri)

        self._centroids = load_as_matrix(self.centroids_uri)
        self._index = read_vector_u64(self.ctx, self.index_uri)

    def query(
        self,
        targets: np.ndarray,
        k=10,
        nqueries=10,
        nthreads=8,
        nprobe=1,
        use_nuv_implementation: bool = False,
    ):
        """
        Query an IVF_FLAT index

        Parameters
        ----------
        targets: numpy.ndarray
            ND Array of query targets
        k: int
            Number of top results to return per target
        nqueries: int
            Number of queries
        nthreads: int
            Number of threads to use for query
        nprobe: int
            number of probes
        use_nuv_implementation: bool
            wether to use the nuv query implementation. Default: False
        """
        assert targets.dtype == np.float32

        targets_m = array_to_matrix(targets)
        if self.memory_budget == -1:
            r = ivf_query_ram(
                self.dtype,
                self._db,
                self._centroids,
                targets_m,
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
                targets_m,
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

        return np.array(r)

    def distributed_query(
        self,
        targets: np.ndarray,
        k=10,
        nthreads=8,
        nprobe=1,
        num_nodes=5,
        mode: Mode = Mode.REALTIME,
    ):
        """
        Distributed Query on top of an IVF_FLAT index

        Parameters
        ----------
        targets: numpy.ndarray
            ND Array of query targets
        k: int
            Number of top results to return per target
        nqueries: int
            Number of queries
        nthreads: int
            Number of threads to use for query
        nprobe: int
            number of probes
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
        ):
            targets_m = array_to_matrix(query_vectors)
            r = dist_qv(
                dtype=dtype,
                parts_uri=parts_uri,
                ids_uri=ids_uri,
                query_vectors=targets_m,
                active_partitions=active_partitions,
                active_queries=active_queries,
                indices=indices,
                k_nn=k_nn,
            )
            results = []
            for q in range(len(r)):
                tmp_results = []
                for j in range(len(r[q])):
                    tmp_results.append(r[q][j])
                results.append(tmp_results)
            return results

        assert targets.dtype == self.dtype
        if mode == Mode.BATCH:
            d = dag.DAG(
                name="vector-query",
                mode=Mode.BATCH,
                max_workers=num_nodes,
            )
        if mode == Mode.REALTIME:
            d = dag.DAG(
                name="vector-query",
                mode=Mode.REALTIME,
                max_workers=num_nodes,
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

        targets_m = array_to_matrix(targets)
        active_partitions, active_queries = partition_ivf_index(
            centroids=self._centroids, query=targets_m, nprobe=nprobe, nthreads=nthreads
        )
        num_parts = len(active_partitions)

        parts_per_node = int(math.ceil(num_parts / num_nodes))
        nodes = []
        for part in range(0, num_parts, parts_per_node):
            part_end = part + parts_per_node
            if part_end > num_parts:
                part_end = num_parts
            nodes.append(
                submit(
                    dist_qv_udf,
                    dtype=self.dtype,
                    parts_uri=self.parts_db_uri,
                    ids_uri=self.ids_uri,
                    query_vectors=targets,
                    active_partitions=np.array(active_partitions)[part:part_end],
                    active_queries=np.array(
                        active_queries[part:part_end], dtype=object
                    ),
                    indices=np.array(self._index),
                    k_nn=k,
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
        for q in range(targets.shape[1]):
            tmp_results = []
            for j in range(k):
                for r in results:
                    if len(r[q]) > j:
                        if r[q][j][0] > 0:
                            tmp_results.append(r[q][j])
            results_per_query.append(sorted(tmp_results, key=lambda t: t[0])[0:k])
        return results_per_query
