import concurrent.futures as futures
import json
import os
import time
from typing import Any, Mapping, Optional

from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import storage_formats

MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
MAX_INT32 = np.iinfo(np.dtype("int32")).max
MAX_FLOAT_32 = np.finfo(np.dtype("float32")).max
DATASET_TYPE = "vector_search"


class Index:

    """
    Open a Vector index

    Parameters
    ----------
    uri: str
        URI of the index
    config: Optional[Mapping[str, Any]]
        config dictionary, defaults to None
    timestamp: int or tuple(int)
        (default None) If int, open the index at a given timestamp.
        If tuple, open at the given start and end timestamps.
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
    ):
        # If the user passes a tiledb python Config object convert to a dictionary
        if isinstance(config, tiledb.Config):
            config = dict(config)

        self.uri = uri
        self.config = config
        self.ctx = Ctx(config)
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
                if timestamp[0] is not None:
                    if timestamp[0] > self.ingestion_timestamps[0]:
                        self.query_base_array = False
                        self.update_array_timestamp = timestamp
                    else:
                        self.history_index = 0
                        self.base_size = self.base_sizes[self.history_index]
                        self.base_array_timestamp = self.ingestion_timestamps[
                            self.history_index
                        ]
                        self.update_array_timestamp = (
                            self.base_array_timestamp + 1,
                            timestamp[1],
                        )
                else:
                    self.history_index = 0
                    self.base_size = self.base_sizes[self.history_index]
                    self.base_array_timestamp = self.ingestion_timestamps[
                        self.history_index
                    ]
                    self.update_array_timestamp = (
                        self.base_array_timestamp + 1,
                        timestamp[1],
                    )
            elif isinstance(timestamp, int):
                self.history_index = 0
                i = 0
                for ingestion_timestamp in self.ingestion_timestamps:
                    if ingestion_timestamp <= timestamp:
                        self.base_array_timestamp = ingestion_timestamp
                        self.history_index = i
                        self.base_size = self.base_sizes[self.history_index]
                    i += 1
                self.update_array_timestamp = (self.base_array_timestamp + 1, timestamp)
            else:
                raise TypeError(
                    "Unexpected argument type for 'timestamp' keyword argument"
                )
        self.thread_executor = futures.ThreadPoolExecutor()

    def query(self, queries: np.ndarray, k, **kwargs):
        if queries.ndim != 2:
            raise TypeError(
                f"Expected queries to have 2 dimensions (i.e. [[...], etc.]), but it had {queries.ndim} dimensions"
            )

        query_dimensions = queries.shape[0] if queries.ndim == 1 else queries.shape[1]
        if query_dimensions != self.get_dimensions():
            raise TypeError(
                f"A query in queries has {query_dimensions} dimensions, but the indexed data had {self.dimensions} dimensions"
            )

        with tiledb.scope_ctx(ctx_or_config=self.config):
            if (
                not tiledb.array_exists(self.updates_array_uri)
                or not self.has_updates()
            ):
                if self.query_base_array:
                    return self.query_internal(queries, k, **kwargs)
                else:
                    return np.full((queries.shape[0], k), MAX_FLOAT_32), np.full(
                        (queries.shape[0], k), MAX_UINT64
                    )

        # Query with updates
        # Perform the queries in parallel
        retrieval_k = 2 * k
        kwargs["nthreads"] = int(os.cpu_count() / 2)
        future = self.thread_executor.submit(
            Index.query_additions,
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
            internal_results_d = np.full((queries.shape[0], k), MAX_FLOAT_32)
            internal_results_i = np.full((queries.shape[0], k), MAX_UINT64)
        addition_results_d, addition_results_i, updated_ids = future.result()

        # Filter updated vectors
        query_id = 0
        for query in internal_results_i:
            res_id = 0
            for res in query:
                if res in updated_ids:
                    internal_results_d[query_id, res_id] = MAX_FLOAT_32
                    internal_results_i[query_id, res_id] = MAX_UINT64
                if (
                    internal_results_d[query_id, res_id] == 0
                    and internal_results_i[query_id, res_id] == 0
                ):
                    internal_results_d[query_id, res_id] = MAX_FLOAT_32
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
                    addition_results_d[query_id, res_id] = MAX_FLOAT_32
                    addition_results_i[query_id, res_id] = MAX_UINT64
                res_id += 1
            query_id += 1

        results_d = np.hstack((internal_results_d, addition_results_d))
        results_i = np.hstack((internal_results_i, addition_results_i))
        sort_index = np.argsort(results_d, axis=1)
        results_d = np.take_along_axis(results_d, sort_index, axis=1)
        results_i = np.take_along_axis(results_i, sort_index, axis=1)
        return results_d[:, 0:k], results_i[:, 0:k]

    @staticmethod
    def query_additions(
        queries: np.ndarray,
        k,
        dtype,
        updates_array_uri,
        nthreads=8,
        timestamp=None,
        config=None,
    ):
        assert queries.dtype == np.float32
        additions_vectors, additions_external_ids, updated_ids = Index.read_additions(
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
    def read_additions(
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

    def get_dimensions(self):
        raise NotImplementedError

    def query_internal(self, queries: np.ndarray, k, **kwargs):
        raise NotImplementedError

    def has_updates(self):
        if "has_updates" in self.group.meta:
            return self.group.meta["has_updates"]
        else:
            return True

    def set_has_updates(self, has_updates: bool = True):
        if not self.group.meta["has_updates"]:
            self.group.close()
            self.group = tiledb.Group(self.uri, "w", ctx=tiledb.Ctx(self.config))
            self.group.meta["has_updates"] = has_updates
            self.group.close()
            self.group = tiledb.Group(self.uri, "r", ctx=tiledb.Ctx(self.config))

    def update(self, vector: np.array, external_id: np.uint64, timestamp: int = None):
        self.set_has_updates()
        updates_array = self.open_updates_array(timestamp=timestamp)
        vectors = np.empty((1), dtype="O")
        vectors[0] = vector
        updates_array[external_id] = {"vector": vectors}
        updates_array.close()
        self.consolidate_update_fragments()

    def update_batch(
        self, vectors: np.ndarray, external_ids: np.array, timestamp: int = None
    ):
        self.set_has_updates()
        updates_array = self.open_updates_array(timestamp=timestamp)
        updates_array[external_ids] = {"vector": vectors}
        updates_array.close()
        self.consolidate_update_fragments()

    def delete(self, external_id: np.uint64, timestamp: int = None):
        self.set_has_updates()
        updates_array = self.open_updates_array(timestamp=timestamp)
        deletes = np.empty((1), dtype="O")
        deletes[0] = np.array([], dtype=self.dtype)
        updates_array[external_id] = {"vector": deletes}
        updates_array.close()
        self.consolidate_update_fragments()

    def delete_batch(self, external_ids: np.array, timestamp: int = None):
        self.set_has_updates()
        updates_array = self.open_updates_array(timestamp=timestamp)
        deletes = np.empty((len(external_ids)), dtype="O")
        for i in range(len(external_ids)):
            deletes[i] = np.array([], dtype=self.dtype)
        updates_array[external_ids] = {"vector": deletes}
        updates_array.close()
        self.consolidate_update_fragments()

    def consolidate_update_fragments(self):
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

    def get_updates_uri(self):
        return self.updates_array_uri

    def open_updates_array(self, timestamp: int = None):
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
                self.group.add(self.updates_array_uri, name=updates_array_name)
                self.group.close()
                self.group = tiledb.Group(self.uri, "r")
            if timestamp is None:
                timestamp = int(time.time() * 1000)
            return tiledb.open(self.updates_array_uri, mode="w", timestamp=timestamp)

    def consolidate_updates(self, retrain_index: bool = False, **kwargs):
        """
        Parameters
        ----------
        retrain_index: bool
            If true, retrain the index. If false, reuse data from the previous index.
            For IVF_FLAT retraining means we will recompute the centroids - when doing so you can
            pass any ingest() arguments used to configure computing centroids and we will use them
            when recomputing the centroids. Otherwise, if false, we will reuse the centroids from
            the previous index.
        """
        from tiledb.vector_search.ingestion import ingest

        fragments_info = tiledb.array_fragments(
            self.updates_array_uri, ctx=tiledb.Ctx(self.config)
        )
        max_timestamp = self.base_array_timestamp
        for fragment_info in fragments_info:
            if fragment_info.timestamp_range[1] > max_timestamp:
                max_timestamp = fragment_info.timestamp_range[1]
        max_timestamp += 1
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
    def delete_index(uri, config):
        with tiledb.scope_ctx(ctx_or_config=config):
            try:
                group = tiledb.Group(uri, "m")
            except tiledb.TileDBError as err:
                message = str(err)
                if "does not exist" in message:
                    return
                else:
                    raise err
            group.delete()

    @staticmethod
    def clear_history(
        uri: str,
        timestamp: int,
        config: Optional[Mapping[str, Any]] = None,
    ):
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(uri, "r")
            storage_version = group.meta.get("storage_version", "0.1")
            if not storage_formats[storage_version]["SUPPORT_TIMETRAVEL"]:
                raise ValueError(
                    f"Time traveling is not supported for index storage_version={storage_version}"
                )
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
                with tiledb.open(updates_array_uri, "m") as A:
                    A.delete_fragments(0, timestamp)

            if index_type == "FLAT":
                db_uri = group[storage_formats[storage_version]["PARTS_ARRAY_NAME"]].uri
                with tiledb.open(db_uri, "m") as A:
                    A.delete_fragments(0, timestamp)
                if storage_formats[storage_version]["IDS_ARRAY_NAME"] in group:
                    ids_uri = group[
                        storage_formats[storage_version]["IDS_ARRAY_NAME"]
                    ].uri
                    with tiledb.open(ids_uri, "m") as A:
                        A.delete_fragments(0, timestamp)
            elif index_type == "IVF_FLAT":
                db_uri = group[storage_formats[storage_version]["PARTS_ARRAY_NAME"]].uri
                centroids_uri = group[
                    storage_formats[storage_version]["CENTROIDS_ARRAY_NAME"]
                ].uri
                index_array_uri = group[
                    storage_formats[storage_version]["INDEX_ARRAY_NAME"]
                ].uri
                ids_uri = group[storage_formats[storage_version]["IDS_ARRAY_NAME"]].uri
                with tiledb.open(db_uri, "m") as A:
                    A.delete_fragments(0, timestamp)
                with tiledb.open(centroids_uri, "m") as A:
                    A.delete_fragments(0, timestamp)
                with tiledb.open(index_array_uri, "m") as A:
                    A.delete_fragments(0, timestamp)
                with tiledb.open(ids_uri, "m") as A:
                    A.delete_fragments(0, timestamp)
            group.close()


def create_metadata(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    index_type: str,
    storage_version: str,
    group_exists: bool = False,
    config: Optional[Mapping[str, Any]] = None,
):
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
