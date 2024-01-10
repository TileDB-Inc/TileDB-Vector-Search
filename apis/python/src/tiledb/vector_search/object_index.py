import base64
import cloudpickle
import numpy as np
import tiledb

from collections import OrderedDict
from typing import Any, Mapping, Optional, Callable
from tiledb.vector_search import flat_index, ivf_flat_index, Index, IVFFlatIndex, FlatIndex, ingest, generate_embeddings
from tiledb.vector_search.storage_formats import STORAGE_VERSION, storage_formats
from tiledb.cloud.dag import Mode

TILEDB_CLOUD_PROTOCOL = 4


class ObjectIndex:
    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
    ):
        self.uri = uri
        self.config = config
        self.timestamp = timestamp
        group = tiledb.Group(uri, "r")
        self.index_type = group.meta["index_type"]
        group.close()
        if self.index_type == "FLAT":
            self.index = FlatIndex(uri=uri, config=config, timestamp=timestamp)
        elif self.index_type == "IVF_FLAT":
            self.index = IVFFlatIndex(uri=uri, config=config, timestamp=timestamp)
        self.object_array_uri = self.index.group.meta["object_array_uri"]
        self.object_metadata_array_uri = self.index.group.meta["object_metadata_array_uri"]
        self.object_id_dim = self.index.group.meta["object_id_dim"]
        self.load_embedding_model_udf_str = self.index.group.meta["load_embedding_model_udf"]
        self.embedding_udf_str = self.index.group.meta["embedding_udf"]
        self.load_embedding_model_udf = cloudpickle.loads(base64.b64decode(self.load_embedding_model_udf_str))
        self.embedding_udf = cloudpickle.loads(base64.b64decode(self.embedding_udf_str))
        self.model = self.load_embedding_model_udf()

    def query(
        self,
        query_objects: np.ndarray,
        k: int,
        metadata_array_cond: str = None,
        return_objects: bool = True,
        return_metadata: bool = True,
        **kwargs,
    ):
        query_embeddings = self.embedding_udf(self.model, query_objects)
        fetch_k = k
        if metadata_array_cond is not None:
            fetch_k = 10*k

        distances, object_ids = self.index.query(queries=query_embeddings, k=fetch_k, **kwargs)
        unique_ids, idx = np.unique(object_ids, return_inverse=True)
        idx = np.reshape(idx, object_ids.shape)

        if metadata_array_cond is not None:
            with tiledb.open(self.object_metadata_array_uri, mode='r', timestamp=self.timestamp) as metadata_array:
                q = metadata_array.query(cond=metadata_array_cond, coords=True)
                filtered_unique_ids = q.multi_index[unique_ids][self.object_id_dim]
                # print(f"filtered_unique_ids: {filtered_unique_ids}")
                filtered_distances = np.zeros((query_embeddings.shape[0], k))
                filtered_object_ids = np.zeros((query_embeddings.shape[0], k))
                for query_id in range(query_embeddings.shape[0]):
                    write_id = 0
                    for result_id in range(fetch_k):
                        if object_ids[query_id, result_id] in filtered_unique_ids:
                            filtered_distances[query_id, write_id] = distances[query_id, result_id]
                            filtered_object_ids[query_id, write_id] = object_ids[query_id, result_id]
                            write_id += 1
                            if write_id >= k:
                                break

            distances = filtered_distances
            object_ids = filtered_object_ids
            unique_ids, idx = np.unique(object_ids, return_inverse=True)
            idx = np.reshape(idx, object_ids.shape)

        object_metadata = None
        if return_metadata:
            with tiledb.open(self.object_metadata_array_uri, mode='r', timestamp=self.timestamp) as metadata_array:
                unique_metadata = metadata_array.multi_index[unique_ids]
                object_metadata = {}
                for attr in unique_metadata.keys():
                    object_metadata[attr] = unique_metadata[attr][idx]

        if return_objects:
            with tiledb.open(self.object_array_uri, mode='r', timestamp=self.timestamp) as object_array:
                unique_objects = object_array.multi_index[unique_ids]
                objects = OrderedDict()
                for attr in unique_objects.keys():
                    objects[attr] = unique_objects[attr][idx]

        if return_objects and return_metadata:
            return distances, objects, object_metadata
        elif return_objects and not return_metadata:
            return distances, objects
        elif not return_objects and return_metadata:
            return distances, object_ids, object_metadata
        elif not return_objects and not return_metadata:
            return distances, object_ids

    def update_index(
        self,
        object_array_timestamp=None,
        index_timestamp: int = None,
        workers: int = -1,
        objects_per_work_item: int = -1,
        max_tasks_per_stage: int= -1,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        mode: Mode = Mode.LOCAL,
        config: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ):
        embeddings_array_name = storage_formats[self.index.storage_version]["INPUT_VECTORS_ARRAY_NAME"]
        embeddings_uri = f"{self.uri}/{embeddings_array_name}"
        external_ids_array_name = storage_formats[self.index.storage_version]["EXTERNAL_IDS_ARRAY_NAME"]
        external_ids_uri = f"{self.uri}/{external_ids_array_name}"
        generate_embeddings(
            object_array_uri=self.object_array_uri,
            embeddings_uri=embeddings_uri,
            external_ids_uri=external_ids_uri,
            dimensions=self.index.dimensions,
            vector_type=self.index.dtype,
            object_id_dim=self.object_id_dim,
            load_embedding_model_udf=self.load_embedding_model_udf_str,
            embedding_udf=self.embedding_udf_str,
            object_array_timestamp=object_array_timestamp,
            index_timestamp=index_timestamp,
            objects_per_work_item=objects_per_work_item,
            max_tasks_per_stage=max_tasks_per_stage,
            workers=workers,
            verbose=verbose,
            trace_id=trace_id,
            mode=mode,
            config=config,
            namespace=namespace,
        )

        with tiledb.open(embeddings_uri, mode='r', timestamp=index_timestamp) as embeddings_array:
            nonempty_object_array_domain = embeddings_array.nonempty_domain()[1]
            size = nonempty_object_array_domain[1] + 1 - nonempty_object_array_domain[0]
        self.index = ingest(
            index_type=self.index_type,
            index_uri=self.uri,
            source_uri=embeddings_uri,
            external_ids_uri=external_ids_uri,
            index_timestamp=index_timestamp,
            size=size,
            storage_version=self.index.storage_version,
            config=self.config,
            **kwargs,
        )


def encode_udf(func: Callable):
    pickledUDF = cloudpickle.dumps(func, protocol=TILEDB_CLOUD_PROTOCOL)
    return base64.b64encode(pickledUDF).decode("ascii")


def create(
    uri: str,
    index_type: str,
    dimensions: int,
    vector_type: np.dtype,
    object_id_dim: str,
    load_embedding_model_udf: Callable,
    embedding_udf: Callable,
    object_array_uri: str = None,
    object_array_schema: tiledb.ArraySchema = None,
    object_metadata_array_uri: str = None,
    object_metadata_array_schema: tiledb.ArraySchema = None,
    embedding_reference_model: str = None,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    **kwargs,
) -> ObjectIndex:
    if index_type == "FLAT":
        index = flat_index.create(
            uri=uri,
            dimensions=dimensions,
            vector_type=vector_type,
            group_exists=False,
            config=config,
            storage_version=storage_version
        )
    elif index_type == "IVF_FLAT":
        index = ivf_flat_index.create(
            uri=uri,
            dimensions=dimensions,
            vector_type=vector_type,
            group_exists=False,
            config=config,
            storage_version=storage_version
        )

    group = tiledb.Group(uri, "w")
    group.meta["object_id_dim"] = object_id_dim
    group.meta["object_array_uri"] = object_array_uri
    group.meta["object_metadata_array_uri"] = object_metadata_array_uri
    group.meta["embedding_udf"] = encode_udf(embedding_udf)
    group.meta["load_embedding_model_udf"] = encode_udf(load_embedding_model_udf)

    embeddings_array_name = storage_formats[index.storage_version]["INPUT_VECTORS_ARRAY_NAME"]
    external_ids_array_name = storage_formats[index.storage_version]["EXTERNAL_IDS_ARRAY_NAME"]
    filters = storage_formats[index.storage_version]["DEFAULT_ATTR_FILTERS"]
    create_embeddings_array(uri, group, dimensions, vector_type, embeddings_array_name, filters)
    create_external_ids_array(uri, group, external_ids_array_name, filters)
    group.close()
    return ObjectIndex(uri, config, **kwargs)


def create_embeddings_array(
    uri: str,
    group: tiledb.Group,
    dimensions: int,
    vector_type: np.dtype,
    array_name: str,
    filters
):
    embeddings_array_uri = f"{uri}/{array_name}"
    if tiledb.array_exists(embeddings_array_uri):
        raise ValueError(f"Array exists {embeddings_array_uri}")
    tile_size = int(flat_index.TILE_SIZE_BYTES / np.dtype(vector_type).itemsize / dimensions)
    embeddings_array_rows_dim = tiledb.Dim(
        name="rows",
        domain=(0, dimensions - 1),
        tile=dimensions,
        dtype=np.dtype(np.int32),
    )
    embeddings_array_cols_dim = tiledb.Dim(
        name="cols",
        domain=(0, flat_index.MAX_INT32),
        tile=tile_size,
        dtype=np.dtype(np.int32),
    )
    embeddings_array_dom = tiledb.Domain(
        embeddings_array_rows_dim, embeddings_array_cols_dim
    )
    embeddings_array_attr = tiledb.Attr(
        name="values", dtype=vector_type, filters=filters
    )
    embeddings_array_schema = tiledb.ArraySchema(
        domain=embeddings_array_dom,
        sparse=False,
        attrs=[embeddings_array_attr],
        cell_order="col-major",
        tile_order="col-major",
    )
    tiledb.Array.create(embeddings_array_uri, embeddings_array_schema)
    group.add(embeddings_array_uri, name=array_name)


def create_external_ids_array(
    uri: str,
    group: tiledb.Group,
    array_name: str,
    filters
):
    external_ids_array_uri = f"{uri}/{array_name}"
    if tiledb.array_exists(external_ids_array_uri):
        raise ValueError(f"Array exists {external_ids_array_uri}")

    ids_array_rows_dim = tiledb.Dim(
        name="rows",
        domain=(0, flat_index.MAX_INT32),
        tile=100000,
        dtype=np.dtype(np.int32),
    )
    ids_array_dom = tiledb.Domain(ids_array_rows_dim)
    ids_attr = tiledb.Attr(
        name="values",
        dtype=np.dtype(np.uint64),
        filters=filters,
    )
    ids_schema = tiledb.ArraySchema(
        domain=ids_array_dom,
        sparse=False,
        attrs=[ids_attr],
        capacity=100000,
        cell_order="col-major",
        tile_order="col-major",
    )
    tiledb.Array.create(external_ids_array_uri, ids_schema)
    group.add(external_ids_array_uri, name=array_name)