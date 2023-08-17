import numpy as np
import tiledb

from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import storage_formats
from typing import Any, Mapping, Optional

MAX_UINT64 = 2 ** 63 - 1


class Index:
    def __init__(
        self,
        uri,
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


    def query(self, targets: np.ndarray, k, **kwargs):
        # TODO merge results based on scores and use higher k to improve retrieval
        updated_ids = set(self.read_updated_ids())
        internal_results = self.query_internal(targets, k, **kwargs)
        if self.update_arrays_uri is None:
            return internal_results
        addition_results = self.query_additions(targets, k)
        merged_results = np.zeros((targets.shape[0], k), dtype=np.uint64)
        query_id = 0
        for query in internal_results:
            res_id = 0
            additional_res_id = 0
            for res in query:
                if res in updated_ids:
                    merged_results[query_id, res_id] = addition_results[query_id, additional_res_id]
                    additional_res_id += 1
                else:
                    merged_results[query_id, res_id] = res
                res_id += 1
            query_id += 1
        return merged_results

    def query_internal(self, targets: np.ndarray, k, **kwargs):
        raise NotImplementedError

    def query_additions(self, targets: np.ndarray, k):
        assert targets.dtype == np.float32

        additions_vectors, additions_external_ids = self.read_additions()
        targets_m = array_to_matrix(np.transpose(targets))
        r = query_vq_heap_pyarray(
            array_to_matrix(np.transpose(additions_vectors).astype(self.dtype)),
            targets_m,
            StdVector_u64(additions_external_ids),
            k,
            8)

        return np.transpose(np.array(r))

    def update(self, vector: np.array, external_id: np.uint64):
        updates_array = self.open_updates_array()
        updates_array[external_id] = vector
        updates_array.close()

    def update_batch(self, vectors: np.ndarray, external_ids: np.array):
        updates_array = self.open_updates_array()
        updates_array[external_ids] = {'vector': vectors}
        updates_array.close()

    def delete(self, external_id: np.uint64):
        updates_array = self.open_updates_array()
        updates_array[external_id] = np.array([], dtype=self.dtype)
        updates_array.close()

    def delete_batch(self, external_ids: np.array):
        updates_array = self.open_updates_array()
        deletes = np.empty((len(external_ids)), dtype='O')
        for i in range(len(external_ids)):
            deletes[i] = np.array([], dtype=self.dtype)
        updates_array[external_ids] = {'vector': deletes}
        updates_array.close()

    def get_updates_uri(self):
        return self.update_arrays_uri

    def print_updates(self):
        updates_array = tiledb.open(self.update_arrays_uri, mode="r")
        print(updates_array[:])

    def read_additions(self) -> (np.ndarray, np.array):
        if self.update_arrays_uri is None:
            return None, None
        updates_array = tiledb.open(self.update_arrays_uri, mode="r")
        q = updates_array.query(attrs=('vector',), coords=True)
        data = q[:]
        additions_filter = [len(item) > 0 for item in data["vector"]]
        return np.vstack(data["vector"][additions_filter]), data["external_id"][additions_filter]

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
                name="external_id", domain=(0, MAX_UINT64), dtype=np.dtype(np.uint64)
            )
            dom = tiledb.Domain(external_id_dim)
            vector_attr = tiledb.Attr(name="vector", dtype=self.dtype, var=True)
            updates_schema = tiledb.ArraySchema(
                domain=dom,
                sparse=True,
                attrs=[vector_attr],
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
        return ingest(
            index_type=self.index_type,
            index_uri=self.uri,
            size=self.size,
            source_uri=self.db_uri,
            external_ids_uri=self.ids_uri,
            updates_uri=self.update_arrays_uri
        )
