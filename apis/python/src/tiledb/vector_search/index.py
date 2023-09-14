import concurrent.futures as futures
import os
import numpy as np
import sys

from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import storage_formats
from typing import Any, Mapping, Optional

MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
MAX_FLOAT_32 = np.finfo(np.dtype("float32")).max


class Index:
    """
    Open a Vector index

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
    ):
        # If the user passes a tiledb python Config object convert to a dictionary
        if isinstance(config, tiledb.Config):
            config = dict(config)

        self.uri = uri
        self.config = config
        self.ctx = Ctx(config)
        self.group = tiledb.Group(self.uri, "r", ctx=tiledb.Ctx(config))
        self.storage_version = self.group.meta.get("storage_version", "0.1")
        self.update_arrays_uri = None
        self.index_version = self.group.meta.get("index_version", "")
        self.thread_executor = futures.ThreadPoolExecutor()

    def query(self, queries: np.ndarray, k, **kwargs):
        if self.update_arrays_uri is None:
            return self.query_internal(queries, k, **kwargs)

        # Query with updates
        # Perform the queries in parallel
        retrieval_k = 2 * k
        kwargs["nthreads"] = int(os.cpu_count() / 2)
        future = self.thread_executor.submit(
            Index.query_additions,
            queries,
            k,
            self.dtype,
            self.update_arrays_uri,
            int(os.cpu_count() / 2),
        )
        internal_results_d, internal_results_i = self.query_internal(
            queries, retrieval_k, **kwargs
        )
        addition_results_d, addition_results_i, updated_ids = future.result()

        # Filter updated vectors
        query_id = 0
        for query in internal_results_i:
            res_id = 0
            for res in query:
                if res in updated_ids:
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
        queries: np.ndarray, k, dtype, update_arrays_uri, nthreads=8
    ):
        assert queries.dtype == np.float32
        additions_vectors, additions_external_ids, updated_ids = Index.read_additions(
            update_arrays_uri
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
    def read_additions(update_arrays_uri) -> (np.ndarray, np.array):
        if update_arrays_uri is None:
            return None, None, np.array([], np.uint64)
        updates_array = tiledb.open(update_arrays_uri, mode="r")
        q = updates_array.query(attrs=("vector",), coords=True)
        data = q[:]
        updates_array.close()
        updated_ids = data["external_id"]
        additions_filter = [len(item) > 0 for item in data["vector"]]
        if len(data["external_id"][additions_filter]) > 0:
            return (
                np.vstack(data["vector"][additions_filter]),
                data["external_id"][additions_filter],
                updated_ids
            )
        else:
            return None, None, updated_ids

    def query_internal(self, queries: np.ndarray, k, **kwargs):
        raise NotImplementedError

    def update(self, vector: np.array, external_id: np.uint64):
        updates_array = self.open_updates_array()
        vectors = np.empty((1), dtype="O")
        vectors[0] = vector
        updates_array[external_id] = {"vector": vectors}
        updates_array.close()
        self.consolidate_update_fragments()

    def update_batch(self, vectors: np.ndarray, external_ids: np.array):
        updates_array = self.open_updates_array()
        updates_array[external_ids] = {"vector": vectors}
        updates_array.close()
        self.consolidate_update_fragments()

    def delete(self, external_id: np.uint64):
        updates_array = self.open_updates_array()
        deletes = np.empty((1), dtype="O")
        deletes[0] = np.array([], dtype=self.dtype)
        updates_array[external_id] = {"vector": deletes}
        updates_array.close()
        self.consolidate_update_fragments()

    def delete_batch(self, external_ids: np.array):
        updates_array = self.open_updates_array()
        deletes = np.empty((len(external_ids)), dtype="O")
        for i in range(len(external_ids)):
            deletes[i] = np.array([], dtype=self.dtype)
        updates_array[external_ids] = {"vector": deletes}
        updates_array.close()
        self.consolidate_update_fragments()

    def consolidate_update_fragments(self):
        fragments_info = tiledb.array_fragments(self.update_arrays_uri)
        if len(fragments_info) > 10:
            tiledb.consolidate(self.update_arrays_uri)
            tiledb.vacuum(self.update_arrays_uri)

    def get_updates_uri(self):
        return self.update_arrays_uri

    def open_updates_array(self):
        if self.update_arrays_uri is None:
            updates_array_name = storage_formats[self.storage_version][
                "UPDATES_ARRAY_NAME"
            ]
            updates_array_uri = f"{self.group.uri}/{updates_array_name}"
            if tiledb.array_exists(updates_array_uri):
                raise RuntimeError(f"Array {updates_array_uri} already exists.")
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
            tiledb.Array.create(updates_array_uri, updates_schema)
            self.group.close()
            self.group = tiledb.Group(self.uri, "w", ctx=tiledb.Ctx(self.config))
            self.group.add(updates_array_uri, name=updates_array_name)
            self.group.close()
            self.group = tiledb.Group(self.uri, "r", ctx=tiledb.Ctx(self.config))
            self.update_arrays_uri = updates_array_uri
        return tiledb.open(self.update_arrays_uri, mode="w")

    def consolidate_updates(self):
        from tiledb.vector_search.ingestion import ingest

        new_index = ingest(
            index_type=self.index_type,
            index_uri=self.uri,
            size=self.size,
            source_uri=self.db_uri,
            external_ids_uri=self.ids_uri,
            updates_uri=self.update_arrays_uri,
        )
        tiledb.Array.delete_array(self.update_arrays_uri)
        self.group.close()
        self.group = tiledb.Group(self.uri, "w", ctx=tiledb.Ctx(self.config))
        self.group.remove(self.update_arrays_uri)
        self.group.close()
        return new_index
