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


    def query(self, queries: np.ndarray, k, **kwargs):
        updated_ids = set(self.read_updated_ids())
        retrieval_k = k
        if len(updated_ids) > 0:
            retrieval_k = 2*k
        internal_results_d, internal_results_i = self.query_internal(queries, retrieval_k, **kwargs)
        if self.update_arrays_uri is None:
            return internal_results_d[:, 0:k], internal_results_i[:, 0:k]

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
        addition_results_d, addition_results_i = self.query_additions(queries, k)
        if addition_results_d is None:
            return internal_results_d[:, 0:k], internal_results_i[:, 0:k]

        query_id = 0
        for query in addition_results_d:
            res_id = 0
            for res in query:
                if addition_results_d[query_id, res_id] == 0 and addition_results_i[query_id, res_id] == 0:
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

    def query_internal(self, queries: np.ndarray, k, **kwargs):
        raise NotImplementedError

    def query_additions(self, queries: np.ndarray, k):
        assert queries.dtype == np.float32
        additions_vectors, additions_external_ids = self.read_additions()
        if additions_vectors is None:
            return None, None
        queries_m = array_to_matrix(np.transpose(queries))
        d, i = query_vq_heap_pyarray(
            array_to_matrix(np.transpose(additions_vectors).astype(self.dtype)),
            queries_m,
            StdVector_u64(additions_external_ids),
            k,
            8)
        return np.transpose(np.array(d)), np.transpose(np.array(i))

    def update(self, vector: np.array, external_id: np.uint64):
        updates_array = self.open_updates_array()
        updates_array[external_id] = vector
        updates_array.close()
        self.consolidate_update_fragments()

    def update_batch(self, vectors: np.ndarray, external_ids: np.array):
        updates_array = self.open_updates_array()
        updates_array[external_ids] = {'vector': vectors}
        updates_array.close()
        self.consolidate_update_fragments()

    def delete(self, external_id: np.uint64):
        updates_array = self.open_updates_array()
        updates_array[external_id] = np.array([], dtype=self.dtype)
        updates_array.close()
        self.consolidate_update_fragments()

    def delete_batch(self, external_ids: np.array):
        updates_array = self.open_updates_array()
        deletes = np.empty((len(external_ids)), dtype='O')
        for i in range(len(external_ids)):
            deletes[i] = np.array([], dtype=self.dtype)
        updates_array[external_ids] = {'vector': deletes}
        updates_array.close()
        self.consolidate_update_fragments()

    def consolidate_update_fragments(self):
        fragments_info = tiledb.array_fragments(self.update_arrays_uri)
        if(len(fragments_info) > 10):
            tiledb.consolidate(self.update_arrays_uri)
            tiledb.vacuum(self.update_arrays_uri)

    def get_updates_uri(self):
        return self.update_arrays_uri

    def read_additions(self) -> (np.ndarray, np.array):
        if self.update_arrays_uri is None:
            return None, None
        updates_array = tiledb.open(self.update_arrays_uri, mode="r")
        q = updates_array.query(attrs=('vector',), coords=True)
        data = q[:]
        additions_filter = [len(item) > 0 for item in data["vector"]]
        if len(data["external_id"][additions_filter]) > 0:
            return np.vstack(data["vector"][additions_filter]), data["external_id"][additions_filter]
        else:
            return None, None
    def read_updated_ids(self) -> np.array:
        if self.update_arrays_uri is None:
            return np.array([], np.uint64)
        updates_array = tiledb.open(self.update_arrays_uri, mode="r")
        q = updates_array.query(attrs=('vector',), coords=True)
        data = q[:]
        return data["external_id"]

    def open_updates_array(self):
        if self.update_arrays_uri is None:
            updates_array_name = storage_formats[self.storage_version]["UPDATES_ARRAY_NAME"]
            updates_array_uri = f"{self.group.uri}/{updates_array_name}"
            if tiledb.array_exists(updates_array_uri):
                raise RuntimeError(f"Array {updates_array_uri} already exists.")
            external_id_dim = tiledb.Dim(
                name="external_id", domain=(0, MAX_UINT64-1), dtype=np.dtype(np.uint64)
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
            updates_uri=self.update_arrays_uri
        )
        tiledb.Array.delete_array(self.update_arrays_uri)
        return new_index
