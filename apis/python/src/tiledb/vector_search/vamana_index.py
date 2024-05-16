import warnings
from typing import Any, Mapping

import numpy as np

from tiledb.cloud.dag import Mode
from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search import index
from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import STORAGE_VERSION
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.storage_formats import validate_storage_version
from tiledb.vector_search.utils import to_temporal_policy

MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
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
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        **kwargs,
    ):
        self.index_type = INDEX_TYPE
        super().__init__(uri=uri, config=config, timestamp=timestamp)
        self.index = vspy.IndexVamana(self.ctx, uri, to_temporal_policy(timestamp))
        self.uri = uri
        self.db_uri = self.group[
            storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]
        ].uri
        self.ids_uri = self.group[
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"]
        ].uri

        schema = tiledb.ArraySchema.load(self.db_uri, ctx=tiledb.Ctx(self.config))
        self.dimensions = self.index.dimensions()

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
        opt_l: Optional[int] = 100,
        mode: Mode = None,
        resource_class: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        num_workers: int = -1,
        **kwargs,
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
            How deep to search. Should be >= k. Defaults to 100.
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
        num_workers: int
            Only relevant for taskgraph based execution.
            If provided, this is the number of workers to use for the query execution.
        """
        print("[vamana_index@query_internal] mode", mode)
        warnings.warn("The Vamana index is not yet supported, please use with caution.")
        if self.size == 0:
            return np.full((queries.shape[0], k), index.MAX_FLOAT_32), np.full(
                (queries.shape[0], k), index.MAX_UINT64
            )

        if mode != Mode.BATCH and resources:
            raise TypeError("Can only pass resources in BATCH mode")
        if (mode != Mode.REALTIME and mode != Mode.BATCH) and resource_class:
            raise TypeError("Can only pass resource_class in REALTIME or BATCH mode")

        assert queries.dtype == np.float32
        if opt_l < k:
            warnings.warn(f"opt_l ({opt_l}) should be >= k ({k}), setting to k")
            opt_l = k

        if queries.ndim == 1:
            queries = np.array([queries])
        queries = np.transpose(queries)
        if not queries.flags.f_contiguous:
            queries = queries.copy(order="F")
        if mode is None:
            print("[vamana_index@query_internal] mode is None")
            queries_feature_vector_array = vspy.FeatureVectorArray(queries)
            distances, ids = self.index.query(queries_feature_vector_array, k, opt_l)
            return np.array(distances, copy=False), np.array(ids, copy=False)
        else:
            print("[vamana_index@query_internal] use self.taskgraph_query()")
            return self.taskgraph_query(
                queries=queries,
                k=k,
                opt_l=opt_l,
                mode=mode,
                resource_class=resource_class,
                resources=resources,
                num_workers=num_workers,
                config=self.config,
            )

    def taskgraph_query(
        self,
        queries: np.ndarray,
        k: int,
        opt_l: int,
        mode: Mode,
        resource_class: Optional[str],
        resources: Optional[Mapping[str, Any]],
        num_workers: int,
        config: Optional[Mapping[str, Any]],
    ):
        """
        Query an VAMANA index using TileDB cloud taskgraphs

        Parameters
        ----------
        queries: numpy.ndarray
            ND Array of queries
        k: int
            Number of top results to return per query
        opt_l: int
            How deep to search. Should be >= k.
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
        num_workers: int
            Only relevant for taskgraph based execution.
            If provided, this is the number of workers to use for the query execution.
        config: None
            config dictionary, defaults to None
        """
        from functools import partial

        import numpy as np

        from tiledb.cloud import dag
        from tiledb.cloud.dag import Mode

        if resource_class and resources:
            raise TypeError("Cannot provide both resource_class and resources")

        def query_udf(
            uri: str,
            queries: np.ndarray,
            opt_l: int,
            k: int,
            config: Optional[Mapping[str, Any]] = None,
            timestamp: int = 0,
        ):
            print("[vamana_index@taskgraph_query@query_udf]")
            from tiledb.vector_search import _tiledbvspy as vspy

            ctx = vspy.Ctx(config)
            index = vspy.IndexVamana(ctx, uri, to_temporal_policy(timestamp))
            queries_feature_vector_array = vspy.FeatureVectorArray(queries)
            distances, ids = index.query(queries_feature_vector_array, k, opt_l)
            return np.array(distances, copy=False), np.array(ids, copy=False)

        assert queries.dtype == np.float32
        if num_workers == -1:
            num_workers = 5
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

        node = submit(
            query_udf,
            uri=self.uri,
            queries=queries,
            opt_l=opt_l,
            k=k,
            config=config,
            timestamp=self.base_array_timestamp,
            resource_class="large"
            if (not resources and not resource_class)
            else resource_class,
            resources=resources,
            image_name="3.9-vectorsearch",
        )

        d.compute()
        d.wait()
        results = node.result()
        print("[vamana_index@taskgraph_query] results", results)
        return results


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    **kwargs,
) -> VamanaIndex:
    warnings.warn("The Vamana index is not yet supported, please use with caution.")
    validate_storage_version(storage_version)
    ctx = vspy.Ctx(config)
    index = vspy.IndexVamana(
        feature_type=np.dtype(vector_type).name,
        id_type=np.dtype(np.uint64).name,
        adjacency_row_index_type=np.dtype(np.uint64).name,
        dimensions=dimensions,
    )
    # TODO(paris): Run all of this with a single C++ call.
    empty_vector = vspy.FeatureVectorArray(
        dimensions, 0, np.dtype(vector_type).name, np.dtype(np.uint64).name
    )
    index.train(empty_vector)
    index.add(empty_vector)
    index.write_index(ctx, uri, vspy.TemporalPolicy(0), storage_version)
    return VamanaIndex(uri=uri, config=config, memory_budget=1000000)
