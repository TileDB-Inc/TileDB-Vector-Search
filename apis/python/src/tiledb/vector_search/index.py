import concurrent.futures as futures
import json
import os
import time
from typing import Any, Mapping, Optional

from tiledb.cloud.dag import Mode
from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.utils import MAX_FLOAT32
from tiledb.vector_search.utils import MAX_UINT64
from tiledb.vector_search.utils import add_to_group
from tiledb.vector_search.utils import is_type_erased_index

DATASET_TYPE = "vector_search"


class Index:
    """
    Abstract Vector Index class.

    All Vector Index algorithm implementations are instantiations of this class. Apart
    from the abstract method interfaces, `Index` provides implementations for common
    tasks i.e. supporting updates, time-traveling and metadata management.

    Opens an `Index` reading metadata and applying time-traveling options.

    Do not use this directly but rather instantiate the concrete Index classes.

    Parameters
    ----------
    uri: str
        URI of the index.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    timestamp: int or tuple(int)
        If int, open the index at a given timestamp.
        If tuple, open at the given start and end timestamps.
    open_for_remote_query_execution: bool
        If `True`, do not load any index data in main memory locally, and instead load index data in the TileDB Cloud taskgraph created when a non-`None` `driver_mode` is passed to `query()`.
        If `False`, load index data in main memory locally. Note that you can still use a taskgraph for query execution, you'll just end up loading the data both on your local machine and in the cloud taskgraph.
    """

    def __init__(
        self,
        uri: str,
        open_for_remote_query_execution: bool,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
    ):
        # If the user passes a tiledb python Config object convert to a dictionary
        if isinstance(config, tiledb.Config):
            config = dict(config)

        self.uri = uri
        self.open_for_remote_query_execution = open_for_remote_query_execution
        self.config = config
        self.ctx = vspy.Ctx(config)
        self.group = tiledb.Group(self.uri, "r", ctx=tiledb.Ctx(config))
        self.storage_version = self.group.meta.get("storage_version", "0.1")
        if (
            not storage_formats[self.storage_version]["SUPPORT_TIMETRAVEL"]
            and timestamp is not None
        ):
            raise ValueError(
                f"Time traveling is not supported for index storage_version={self.storage_version}"
            )
        updates_array_name = storage_formats[self.storage_version]["UPDATES_ARRAY_NAME"]
        if updates_array_name in self.group:
            self.updates_array_uri = self.group[
                storage_formats[self.storage_version]["UPDATES_ARRAY_NAME"]
            ].uri
        else:
            self.updates_array_uri = f"{self.group.uri}/{updates_array_name}"
        self.index_version = self.group.meta.get("index_version", "")
        self.ingestion_timestamps = [
            int(x)
            for x in list(json.loads(self.group.meta.get("ingestion_timestamps", "[]")))
        ]
        self.history_index = len(self.ingestion_timestamps) - 1
        if self.history_index > -1:
            self.latest_ingestion_timestamp = self.ingestion_timestamps[
                self.history_index
            ]
        else:
            self.latest_ingestion_timestamp = MAX_UINT64
        self.base_sizes = [
            int(x) for x in list(json.loads(self.group.meta.get("base_sizes", "[]")))
        ]
        if len(self.base_sizes) > 0:
            self.base_size = self.base_sizes[self.history_index]
        else:
            self.base_size = -1
        self.base_array_timestamp = self.latest_ingestion_timestamp
        self.query_base_array = True
        self.update_array_timestamp = (self.base_array_timestamp + 1, None)
        if timestamp is None:
            self.base_array_timestamp = 0
        else:
            if isinstance(timestamp, tuple):
                if len(timestamp) != 2:
                    raise ValueError(
                        "'timestamp' argument expects either int or tuple(start: int, end: int)"
                    )
                if (
                    timestamp[0] is not None
                    and timestamp[0] > self.ingestion_timestamps[0]
                ):
                    self.query_base_array = False
                    self.update_array_timestamp = timestamp
                else:
                    if (
                        timestamp[1] is None
                        or timestamp[1] >= self.ingestion_timestamps[0]
                    ):
                        self.history_index = 0
                        self.base_size = self.base_sizes[self.history_index]
                        self.base_array_timestamp = self.ingestion_timestamps[
                            self.history_index
                        ]
                    else:
                        # If the timestamp is before the first ingestion, we'll have no vectors to return.
                        self.history_index = 0
                        self.base_size = 0
                        self.base_array_timestamp = timestamp[1]
                        self.query_base_array = False

                    self.update_array_timestamp = (
                        self.base_array_timestamp + 1,
                        timestamp[1],
                    )

            elif isinstance(timestamp, int):
                # NOTE(paris): We could instead use the same logic as in the else statement above,
                # but we do it like this as a performance improvment so that we read less from the
                # updates array and more from ingestions. Above we need to read just the first
                # ingestion and then from the updates array in case we get a timestamp in between an
                # ingestion and an update.
                if timestamp >= self.ingestion_timestamps[0]:
                    self.history_index = 0
                    i = 0
                    for ingestion_timestamp in self.ingestion_timestamps:
                        if ingestion_timestamp <= timestamp:
                            self.base_array_timestamp = ingestion_timestamp
                            self.history_index = i
                            self.base_size = self.base_sizes[self.history_index]
                        i += 1
                else:
                    # If the timestamp is before the first ingestion, we'll have no vectors to return.
                    self.history_index = 0
                    self.base_size = 0
                    self.base_array_timestamp = timestamp
                    self.query_base_array = False

                self.update_array_timestamp = (self.base_array_timestamp + 1, timestamp)
            else:
                raise TypeError(
                    "Unexpected argument type for 'timestamp' keyword argument"
                )
        self.thread_executor = futures.ThreadPoolExecutor()
        self.has_updates = self._check_has_updates()

    def _query_with_driver(
        self,
        queries: np.ndarray,
        k: int,
        driver_mode=None,
        driver_resources=None,
        driver_access_credentials_name=None,
        **kwargs,
    ):
        from tiledb.cloud import dag

        def query_udf(index_type, index_open_kwargs, query_kwargs):
            from tiledb.vector_search.flat_index import FlatIndex
            from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
            from tiledb.vector_search.vamana_index import VamanaIndex

            # Open index
            if index_type == "FLAT":
                index = FlatIndex(**index_open_kwargs)
            elif index_type == "IVF_FLAT":
                index = IVFFlatIndex(**index_open_kwargs)
            elif index_type == "VAMANA":
                index = VamanaIndex(**index_open_kwargs)

            # Query index
            return index.query(**query_kwargs)

        d = dag.DAG(
            name="vector-query",
            mode=driver_mode,
            max_workers=1,
        )
        query_kwargs = {
            "queries": queries,
            "k": k,
        }
        query_kwargs.update(kwargs)
        node = d.submit(
            query_udf,
            self.index_type,
            self.index_open_kwargs,
            query_kwargs,
            name="vector-query-driver",
            resources=driver_resources,
            image_name="vectorsearch",
            access_credentials_name=driver_access_credentials_name,
        )
        d.compute()
        d.wait()
        return node.result()

    def query(
        self,
        queries: np.ndarray,
        k: int,
        driver_mode: Mode = None,
        driver_resources: Optional[str] = None,
        driver_access_credentials_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Queries an index with a set of query vectors, retrieving the `k` most similar vectors for each query.

        This provides an algorithm-agnostic implementation for updates:

        - Queries the non-consolidated updates table.
        - Calls the algorithm specific implementation of `query_internal` to query the base data.
        - Merges the results applying the updated data.

        You can control where the query is executed by setting the `driver_mode` parameter:
        - With `driver_mode = None`, the driver logic for the query will be executed locally.
        - If `driver_mode` is not `None`, we will use a TileDB cloud taskgraph to re-open the index and run the query.
        With both options, certain implementations, i.e. IVF Flat, may let you create further TileDB taskgraphs as defined in the implementation specific `query_internal` methods.

        Parameters
        ----------
        queries: np.ndarray
            2D array of query vectors. This can be used as a batch query interface by passing multiple queries in one call.
        k: int
            Number of results to return per query vector.
        driver_mode: Mode
            If not `None`, the query will be executed in a TileDB cloud taskgraph using the driver mode specified.
        driver_resources: Optional[str]
            If `driver_mode` was not `None`, the resources to use for the driver execution.
        driver_access_credentials_name: Optional[str]
            If `driver_mode` was not `None`, the access credentials name to use for the driver execution.
        **kwargs
            Extra kwargs passed here are passed to the `query_internal` implementation of the concrete index class.
        """
        if queries.ndim != 2:
            raise TypeError(
                f"Expected queries to have 2 dimensions (i.e. [[...], etc.]), but it had {queries.ndim} dimensions"
            )

        query_dimensions = queries.shape[0] if queries.ndim == 1 else queries.shape[1]
        if query_dimensions != self.get_dimensions():
            raise TypeError(
                f"A query in queries has {query_dimensions} dimensions, but the indexed data had {self.dimensions} dimensions"
            )

        if queries.dtype != np.float32:
            raise TypeError(
                f"Expected queries to have dtype np.float32, but it had dtype {queries.dtype}"
            )

        if driver_mode == Mode.LOCAL:
            # @todo: Fix bug with driver_mode=Mode.LOCAL and remove this check.
            raise TypeError(
                "Cannot pass driver_mode=Mode.LOCAL to query() - use driver_mode=None to query locally."
            )

        if driver_mode is not None:
            return self._query_with_driver(
                queries,
                k,
                driver_mode,
                driver_resources,
                driver_access_credentials_name,
                **kwargs,
            )

        if self.open_for_remote_query_execution:
            raise ValueError(
                "Cannot query an index with driver_mode=None without loading the index data in main memory. Set open_for_remote_query_execution=False when creating the index to load the index data before query."
            )

        with tiledb.scope_ctx(ctx_or_config=self.config):
            if not self.has_updates:
                if self.query_base_array:
                    return self.query_internal(queries, k, **kwargs)
                else:
                    return np.full((queries.shape[0], k), MAX_FLOAT32), np.full(
                        (queries.shape[0], k), MAX_UINT64
                    )

        # Query with updates
        # Perform the queries in parallel
        retrieval_k = 2 * k
        kwargs["nthreads"] = int(os.cpu_count() / 2)
        future = self.thread_executor.submit(
            Index._query_additions,
            queries,
            k,
            self.dtype,
            self.updates_array_uri,
            int(os.cpu_count() / 2),
            self.update_array_timestamp,
            self.config,
        )
        if self.query_base_array:
            internal_results_d, internal_results_i = self.query_internal(
                queries, retrieval_k, **kwargs
            )
        else:
            internal_results_d = np.full((queries.shape[0], k), MAX_FLOAT32)
            internal_results_i = np.full((queries.shape[0], k), MAX_UINT64)
        addition_results_d, addition_results_i, updated_ids = future.result()

        # Filter updated vectors
        query_id = 0
        for query in internal_results_i:
            res_id = 0
            for res in query:
                if res in updated_ids:
                    internal_results_d[query_id, res_id] = MAX_FLOAT32
                    internal_results_i[query_id, res_id] = MAX_UINT64
                res_id += 1
            query_id += 1
        sort_index = np.argsort(internal_results_d, axis=1)
        internal_results_d = np.take_along_axis(internal_results_d, sort_index, axis=1)
        internal_results_i = np.take_along_axis(internal_results_i, sort_index, axis=1)

        # Merge update results
        if addition_results_d is None:
            return internal_results_d[:, 0:k], internal_results_i[:, 0:k]

        query_id = 0
        for query in addition_results_d:
            res_id = 0
            for res in query:
                if (
                    addition_results_d[query_id, res_id] == 0
                    and addition_results_i[query_id, res_id] == 0
                ):
                    addition_results_d[query_id, res_id] = MAX_FLOAT32
                    addition_results_i[query_id, res_id] = MAX_UINT64
                res_id += 1
            query_id += 1

        results_d = np.hstack((internal_results_d, addition_results_d))
        results_i = np.hstack((internal_results_i, addition_results_i))
        sort_index = np.argsort(results_d, axis=1)
        results_d = np.take_along_axis(results_d, sort_index, axis=1)
        results_i = np.take_along_axis(results_i, sort_index, axis=1)
        return results_d[:, 0:k], results_i[:, 0:k]

    def update(self, vector: np.array, external_id: np.uint64, timestamp: int = None):
        """
        Updates a `vector` by its `external_id`.

        This can be used to add new vectors or update an existing vector with the same `external_id`.

        Parameters
        ----------
        vector: np.array
            Vector data to be updated.
        external_id: np.uint64
            External ID of the vector.
        timestamp: int
            Timestamp to use for the update to take place at.
        """
        self._set_has_updates()
        updates_array = self._open_updates_array(timestamp=timestamp)
        vectors = np.empty((1), dtype="O")
        vectors[0] = vector
        updates_array[external_id] = {"vector": vectors}
        updates_array.close()
        self._consolidate_update_fragments()

    def update_batch(
        self, vectors: np.ndarray, external_ids: np.array, timestamp: int = None
    ):
        """
        Updates a set `vectors` by their `external_ids`.

        This can be used to add new vectors or update existing vectors with the same `external_id`.

        Parameters
        ----------
        vectors: np.ndarray
            2D array containing the vectors to be updated.
        external_ids: np.uint64
            External IDs of the vectors.
        timestamp: int
            Timestamp to use for the updates to take place at.
        """
        self._set_has_updates()
        updates_array = self._open_updates_array(timestamp=timestamp)
        updates_array[external_ids] = {"vector": vectors}
        updates_array.close()
        self._consolidate_update_fragments()

    def delete(self, external_id: np.uint64, timestamp: int = None):
        """
        Deletes a vector by its `external_id`.

        Parameters
        ----------
        external_id: np.uint64
            External ID of the vector to be deleted.
        timestamp: int
            Timestamp to use for the deletes to take place at.
        """
        self._set_has_updates()
        updates_array = self._open_updates_array(timestamp=timestamp)
        deletes = np.empty((1), dtype="O")
        deletes[0] = np.array([], dtype=self.dtype)
        updates_array[external_id] = {"vector": deletes}
        updates_array.close()
        self._consolidate_update_fragments()

    def delete_batch(self, external_ids: np.array, timestamp: int = None):
        """
        Deletes vectors by their `external_ids`.

        Parameters
        ----------
        external_ids: np.array
            External IDs of the vectors to be deleted.
        timestamp: int
            Timestamp to use for the deletes to take place at.
        """
        self._set_has_updates()
        updates_array = self._open_updates_array(timestamp=timestamp)
        deletes = np.empty((len(external_ids)), dtype="O")
        for i in range(len(external_ids)):
            deletes[i] = np.array([], dtype=self.dtype)
        updates_array[external_ids] = {"vector": deletes}
        updates_array.close()
        self._consolidate_update_fragments()

    def consolidate_updates(self, retrain_index: bool = False, **kwargs):
        """
        Consolidates updates by merging updates form the updates table into the base index.

        The consolidation process is used to avoid query latency degradation as more updates
        are added to the index. It triggers a base index re-indexing, merging the non-consolidated
        updates and the rest of the base vectors.

        Parameters
        ----------
        retrain_index: bool
            If true, retrain the index. If false, reuse data from the previous index.
            For IVF_FLAT retraining means we will recompute the centroids - when doing so you can
            pass any ingest() arguments used to configure computing centroids and we will use them
            when recomputing the centroids. Otherwise, if false, we will reuse the centroids from
            the previous index.
        **kwargs
            Extra kwargs passed here are passed to `ingest` function.
        """
        from tiledb.vector_search.ingestion import ingest

        if self.index_type == "IVF_PQ":
            # TODO(SC-48888): Fix consolidation for IVF_PQ.
            raise ValueError("IVF_PQ indexes do not support consolidation yet.")

        fragments_info = tiledb.array_fragments(
            self.updates_array_uri, ctx=tiledb.Ctx(self.config)
        )
        max_timestamp = self.base_array_timestamp
        for fragment_info in fragments_info:
            if fragment_info.timestamp_range[1] > max_timestamp:
                max_timestamp = fragment_info.timestamp_range[1]
        max_timestamp += 1
        # Consolidate all updates since the previous ingestion_timestamp.
        # This is a performance optimization. We skip this for remote arrays as consolidation
        # of remote arrays currently only supports modes `fragment_meta, commits, metadata`.
        if not self.updates_array_uri.startswith("tiledb://"):
            conf = tiledb.Config(self.config)
            conf["sm.consolidation.timestamp_start"] = self.latest_ingestion_timestamp
            conf["sm.consolidation.timestamp_end"] = max_timestamp
            tiledb.consolidate(self.updates_array_uri, config=conf)
            tiledb.vacuum(self.updates_array_uri, config=conf)

        # We don't copy the centroids if self.partitions=0 because this means our index was previously empty.
        should_pass_copy_centroids_uri = (
            self.index_type == "IVF_FLAT" and not retrain_index and self.partitions > 0
        )
        if should_pass_copy_centroids_uri:
            # Make sure the user didn't pass an incorrect number of partitions.
            if "partitions" in kwargs and self.partitions != kwargs["partitions"]:
                raise ValueError(
                    f"The passed partitions={kwargs['partitions']} is different than the number of partitions ({self.partitions}) from when the index was created - this is an issue because with retrain_index=True, the partitions from the previous index will be used; to fix, set retrain_index=False, don't pass partitions, or pass the correct number of partitions."
                )
            # We pass partitions through kwargs so that we don't pass it twice.
            kwargs["partitions"] = self.partitions

        new_index = ingest(
            index_type=self.index_type,
            index_uri=self.uri,
            size=self.size,
            source_uri=self.db_uri,
            external_ids_uri=self.ids_uri,
            external_ids_type="TILEDB_ARRAY",
            updates_uri=self.updates_array_uri,
            index_timestamp=max_timestamp,
            storage_version=self.storage_version,
            copy_centroids_uri=self.centroids_uri
            if should_pass_copy_centroids_uri
            else None,
            config=self.config,
            **kwargs,
        )
        return new_index

    @staticmethod
    def delete_index(
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
    ):
        """
        Deletes an index from storage based on its URI.

        Parameters
        ----------
        uri: str
            URI of the index.
        config: Optional[Mapping[str, Any]]
            TileDB config dictionary.
        """
        with tiledb.scope_ctx(ctx_or_config=config):
            try:
                group = tiledb.Group(uri, "m")
            except tiledb.TileDBError as err:
                message = str(err)
                if "does not exist" in message:
                    return
                else:
                    raise err
            group.delete(recursive=True)

    @staticmethod
    def clear_history(
        uri: str,
        timestamp: int,
        config: Optional[Mapping[str, Any]] = None,
    ):
        """
        Clears the history maintained in a Vector Index based on its URI.

        This clears the update history before the provided `timestamp`.

        Use this in collaboration with `consolidate_updates` to periodically cleanup update history.

        Parameters
        ----------
        uri: str
            URI of the index.
        timestamp: int
            Clears update history before this timestamp.
        """
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(uri, "r")
            index_type = group.meta.get("index_type", "")
            storage_version = group.meta.get("storage_version", "0.1")
            if not storage_formats[storage_version]["SUPPORT_TIMETRAVEL"]:
                raise ValueError(
                    f"Time traveling is not supported for index storage_version={storage_version}"
                )

            if is_type_erased_index(index_type):
                if storage_formats[storage_version]["UPDATES_ARRAY_NAME"] in group:
                    updates_array_uri = group[
                        storage_formats[storage_version]["UPDATES_ARRAY_NAME"]
                    ].uri
                    tiledb.Array.delete_fragments(updates_array_uri, 0, timestamp)
                ctx = vspy.Ctx(config)
                if index_type == "VAMANA":
                    vspy.IndexVamana.clear_history(ctx, uri, timestamp)
                elif index_type == "IVF_PQ":
                    vspy.IndexIVFPQ.clear_history(ctx, uri, timestamp)
                else:
                    raise ValueError(f"Unsupported index_type: {index_type}")
                return

            ingestion_timestamps = [
                int(x)
                for x in list(json.loads(group.meta.get("ingestion_timestamps", "[]")))
            ]
            base_sizes = [
                int(x) for x in list(json.loads(group.meta.get("base_sizes", "[]")))
            ]
            partition_history = [
                int(x)
                for x in list(json.loads(group.meta.get("partition_history", "[]")))
            ]
            new_ingestion_timestamps = []
            new_base_sizes = []
            new_partition_history = []
            i = 0
            for ingestion_timestamp in ingestion_timestamps:
                if ingestion_timestamp > timestamp:
                    new_ingestion_timestamps.append(ingestion_timestamp)
                    new_base_sizes.append(base_sizes[i])
                    new_partition_history.append(partition_history[i])
                i += 1
            if len(new_ingestion_timestamps) == 0:
                new_ingestion_timestamps = [0]
                new_base_sizes = [0]
                new_partition_history = [0]
            index_type = group.meta.get("index_type", "")
            group.close()

            group = tiledb.Group(uri, "w")
            group.meta["ingestion_timestamps"] = json.dumps(new_ingestion_timestamps)
            group.meta["base_sizes"] = json.dumps(new_base_sizes)
            group.meta["partition_history"] = json.dumps(new_partition_history)
            group.close()

            group = tiledb.Group(uri, "r")
            if storage_formats[storage_version]["UPDATES_ARRAY_NAME"] in group:
                updates_array_uri = group[
                    storage_formats[storage_version]["UPDATES_ARRAY_NAME"]
                ].uri
                tiledb.Array.delete_fragments(updates_array_uri, 0, timestamp)

            if index_type == "FLAT":
                db_uri = group[storage_formats[storage_version]["PARTS_ARRAY_NAME"]].uri
                tiledb.Array.delete_fragments(db_uri, 0, timestamp)
                if storage_formats[storage_version]["IDS_ARRAY_NAME"] in group:
                    ids_uri = group[
                        storage_formats[storage_version]["IDS_ARRAY_NAME"]
                    ].uri
                    tiledb.Array.delete_fragments(ids_uri, 0, timestamp)
            elif index_type == "IVF_FLAT":
                db_uri = group[storage_formats[storage_version]["PARTS_ARRAY_NAME"]].uri
                centroids_uri = group[
                    storage_formats[storage_version]["CENTROIDS_ARRAY_NAME"]
                ].uri
                index_array_uri = group[
                    storage_formats[storage_version]["INDEX_ARRAY_NAME"]
                ].uri
                ids_uri = group[storage_formats[storage_version]["IDS_ARRAY_NAME"]].uri
                tiledb.Array.delete_fragments(db_uri, 0, timestamp)
                tiledb.Array.delete_fragments(centroids_uri, 0, timestamp)
                tiledb.Array.delete_fragments(index_array_uri, 0, timestamp)
                tiledb.Array.delete_fragments(ids_uri, 0, timestamp)
            else:
                raise ValueError(f"Unsupported index_type: {index_type}")
            group.close()

    def get_dimensions(self):
        """
        Abstract method implemented by all Vector Index implementations.

        Returns the dimension of the vectors in the index.
        """
        raise NotImplementedError

    def query_internal(self, queries: np.ndarray, k: int, **kwargs):
        """
        Abstract method implemented by all Vector Index implementations.

        Queries the base index with a set of query vectors, retrieving the `k` most similar vectors for each query.

        Parameters
        ----------
        queries: np.ndarray
            2D array of query vectors. This can be used as a batch query interface by passing multiple queries in one call.
        k: int
            Number of results to return per query vector.
        **kwargs
            Extra kwargs passed here for each algorithm implementation.
        """
        raise NotImplementedError

    @staticmethod
    def _query_additions(
        queries: np.ndarray,
        k,
        dtype,
        updates_array_uri,
        nthreads=8,
        timestamp=None,
        config=None,
    ):
        additions_vectors, additions_external_ids, updated_ids = Index._read_additions(
            updates_array_uri, timestamp, config
        )
        if additions_vectors is None:
            return None, None, updated_ids

        queries_m = array_to_matrix(np.transpose(queries))
        d, i = query_vq_heap_pyarray(
            array_to_matrix(np.transpose(additions_vectors).astype(dtype)),
            queries_m,
            StdVector_u64(additions_external_ids),
            k,
            nthreads,
        )
        return np.transpose(np.array(d)), np.transpose(np.array(i)), updated_ids

    @staticmethod
    def _read_additions(
        updates_array_uri, timestamp=None, config=None
    ) -> (np.ndarray, np.array):
        with tiledb.scope_ctx(ctx_or_config=config):
            if updates_array_uri is None:
                return None, None, np.array([], np.uint64)
            updates_array = tiledb.open(
                updates_array_uri, mode="r", timestamp=timestamp
            )
            q = updates_array.query(attrs=("vector",), coords=True)
            data = q[:]
            updates_array.close()
            updated_ids = data["external_id"]
            additions_filter = [len(item) > 0 for item in data["vector"]]
            if len(data["external_id"][additions_filter]) > 0:
                return (
                    np.vstack(data["vector"][additions_filter]),
                    data["external_id"][additions_filter],
                    updated_ids,
                )
            else:
                return None, None, updated_ids

    def _check_has_updates(self):
        with tiledb.scope_ctx(ctx_or_config=self.config):
            array_exists = tiledb.array_exists(self.updates_array_uri)
        if "has_updates" in self.group.meta:
            has_updates = self.group.meta["has_updates"]
        else:
            has_updates = True
        return array_exists and has_updates

    def _set_has_updates(self, has_updates: bool = True):
        self.has_updates = has_updates
        if (
            "has_updates" not in self.group.meta
            or self.group.meta["has_updates"] != has_updates
        ):
            self.group.close()
            self.group = tiledb.Group(self.uri, "w", ctx=tiledb.Ctx(self.config))
            self.group.meta["has_updates"] = has_updates
            self.group.close()
            self.group = tiledb.Group(self.uri, "r", ctx=tiledb.Ctx(self.config))

    def _consolidate_update_fragments(self):
        with tiledb.scope_ctx(ctx_or_config=self.config):
            fragments_info = tiledb.array_fragments(self.updates_array_uri)
        count_fragments = 0
        for timestamp_range in fragments_info.timestamp_range:
            if timestamp_range[1] > self.latest_ingestion_timestamp:
                count_fragments += 1
        if count_fragments > 10:
            conf = tiledb.Config(self.config)
            conf["sm.consolidation.timestamp_start"] = self.latest_ingestion_timestamp
            tiledb.consolidate(self.updates_array_uri, config=conf)
            tiledb.vacuum(self.updates_array_uri, config=conf)

    def _open_updates_array(self, timestamp: int = None):
        with tiledb.scope_ctx(ctx_or_config=self.config):
            if timestamp is not None:
                if not storage_formats[self.storage_version]["SUPPORT_TIMETRAVEL"]:
                    raise ValueError(
                        f"Time traveling is not supported for index storage_version={self.storage_version}"
                    )
                if timestamp <= self.latest_ingestion_timestamp:
                    raise ValueError(
                        f"Updates at a timestamp before the latest_ingestion_timestamp are not supported. "
                        f"timestamp: {timestamp}, latest_ingestion_timestamp: {self.latest_ingestion_timestamp}"
                    )
            if not tiledb.array_exists(self.updates_array_uri):
                updates_array_name = storage_formats[self.storage_version][
                    "UPDATES_ARRAY_NAME"
                ]
                external_id_dim = tiledb.Dim(
                    name="external_id",
                    domain=(0, MAX_UINT64 - 1),
                    dtype=np.dtype(np.uint64),
                )
                dom = tiledb.Domain(external_id_dim)
                vector_attr = tiledb.Attr(name="vector", dtype=self.dtype, var=True)
                updates_schema = tiledb.ArraySchema(
                    domain=dom,
                    sparse=True,
                    attrs=[vector_attr],
                    allows_duplicates=False,
                )
                tiledb.Array.create(self.updates_array_uri, updates_schema)
                self.group.close()
                self.group = tiledb.Group(self.uri, "w")
                add_to_group(self.group, self.updates_array_uri, updates_array_name)
                self.group.close()
                self.group = tiledb.Group(self.uri, "r")
            if timestamp is None:
                timestamp = int(time.time() * 1000)
            return tiledb.open(self.updates_array_uri, mode="w", timestamp=timestamp)


def create_metadata(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    index_type: str,
    storage_version: str,
    group_exists: bool = False,
    config: Optional[Mapping[str, Any]] = None,
):
    """
    Creates the index group adding index metadata.
    """
    with tiledb.scope_ctx(ctx_or_config=config):
        if not group_exists:
            try:
                tiledb.group_create(uri)
            except tiledb.TileDBError as err:
                raise err
        group = tiledb.Group(uri, "w")
        group.meta["dataset_type"] = DATASET_TYPE
        group.meta["dtype"] = np.dtype(vector_type).name
        group.meta["storage_version"] = storage_version
        group.meta["index_type"] = index_type
        group.meta["base_sizes"] = json.dumps([0])
        group.meta["ingestion_timestamps"] = json.dumps([0])
        group.meta["has_updates"] = False
        group.close()
