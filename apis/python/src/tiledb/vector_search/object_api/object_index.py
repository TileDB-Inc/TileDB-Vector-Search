import base64
import cloudpickle
import numpy as np
import tiledb

from collections import OrderedDict
from typing import Any, Mapping, Optional, Dict
from tiledb.vector_search import flat_index, ivf_flat_index, Index, IVFFlatIndex, FlatIndex, ingest
from tiledb.vector_search.storage_formats import STORAGE_VERSION, storage_formats
from tiledb.cloud.dag import Mode
from tiledb.vector_search.object_readers import ObjectReader
from tiledb.vector_search.embeddings import ObjectEmbedding
import tiledb.vector_search.object_api as object_api

TILEDB_CLOUD_PROTOCOL = 4


class ObjectIndex:
    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        load_embedding: bool = True,
        **kwargs
    ):
        with tiledb.scope_ctx(ctx_or_config=config):
            self.uri = uri
            self.config = config
            self.timestamp = timestamp
            group = tiledb.Group(uri, "r")
            self.index_type = group.meta["index_type"]
            group.close()
            if self.index_type == "FLAT":
                self.index = FlatIndex(uri=uri, config=config, timestamp=timestamp, **kwargs)
            elif self.index_type == "IVF_FLAT":
                self.index = IVFFlatIndex(uri=uri, config=config, timestamp=timestamp, **kwargs)

            self.object_reader_source_code = self.index.group.meta["object_reader_source_code"]
            self.object_reader_class_name = self.index.group.meta["object_reader_class_name"]
            self.object_reader_kwargs = cloudpickle.loads(base64.b64decode(self.index.group.meta["object_reader_kwargs"]))
            self.object_reader = instantiate_object(
                code = self.object_reader_source_code,
                class_name = self.object_reader_class_name,
                **self.object_reader_kwargs
            )
            self.embedding_source_code = self.index.group.meta["embedding_source_code"]
            self.embedding_class_name = self.index.group.meta["embedding_class_name"]
            self.embedding_kwargs = cloudpickle.loads(base64.b64decode(self.index.group.meta["embedding_kwargs"]))
            self.embedding = instantiate_object(
                code = self.embedding_source_code,
                class_name = self.embedding_class_name,
                **self.embedding_kwargs
            )
            self.embedding_loaded = False
            if load_embedding:
                self.embedding.load()
                self.embedding_loaded = True

            self.object_metadata_array_uri = self.index.group.meta["object_metadata_array_uri"]
            self.materialize_object_metadata = self.index.group.meta["materialize_object_metadata"]
            self.object_metadata_external_id_dim = self.index.group.meta["object_metadata_external_id_dim"]

    def query(
        self,
        query_objects: np.ndarray,
        k: int,
        query_metadata: OrderedDict = None,
        metadata_array_cond: str = None,
        return_objects: bool = True,
        return_metadata: bool = True,
        **kwargs,
    ):
        if metadata_array_cond is not None and self.object_metadata_array_uri is None:
            raise AttributeError("metadata_array_cond can't be applied when there is no metadata array") 

        if not self.embedding_loaded:
            self.embedding.load()
            self.embedding_loaded = True
        query_embeddings = self.embedding.embed(objects=query_objects, metadata=query_metadata)
        fetch_k = k
        if metadata_array_cond is not None:
            fetch_k = min(50*k, self.index.size) 

        distances, object_ids = self.index.query(queries=query_embeddings, k=fetch_k, **kwargs)
        unique_ids, idx = np.unique(object_ids, return_inverse=True)
        idx = np.reshape(idx, object_ids.shape)

        if metadata_array_cond is not None:
            with tiledb.open(self.object_metadata_array_uri, mode='r', timestamp=self.timestamp, config=self.config) as metadata_array:
                q = metadata_array.query(cond=metadata_array_cond, coords=True)
                filtered_unique_ids = q.multi_index[unique_ids][self.object_metadata_external_id_dim]
                filtered_distances = np.zeros((query_embeddings.shape[0], k)).astype(object_ids.dtype)
                filtered_object_ids = np.zeros((query_embeddings.shape[0], k)).astype(object_ids.dtype)
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
            with tiledb.open(self.object_metadata_array_uri, mode='r', timestamp=self.timestamp, config=self.config) as metadata_array:
                unique_metadata = metadata_array.multi_index[unique_ids]
                object_metadata = {}
                for attr in unique_metadata.keys():
                    object_metadata[attr] = unique_metadata[attr][idx]

        if return_objects:
            unique_objects = self.object_reader.read_objects_by_external_ids(unique_ids)
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
        worker_resources: Dict = None,
        objects_per_partition: int = -1,
        object_partitions_per_task: int = -1,
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
        metadata_array_uri = None
        if self.materialize_object_metadata:
            metadata_array_uri = self.object_metadata_array_uri
        if config is None:
            config = self.config
        object_api.ingest_embeddings(
            object_index=self,
            embeddings_uri=embeddings_uri,
            external_ids_uri=external_ids_uri,
            metadata_array_uri=metadata_array_uri,
            index_timestamp=index_timestamp,
            objects_per_partition=objects_per_partition,
            object_partitions_per_task=object_partitions_per_task,
            max_tasks_per_stage=max_tasks_per_stage,
            workers=workers,
            worker_resources=worker_resources,
            verbose=verbose,
            trace_id=trace_id,
            mode=mode,
            config=config,
            namespace=namespace,
        )

        with tiledb.open(embeddings_uri, mode='r', timestamp=index_timestamp, config=config) as embeddings_array:
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
            config=config,
            mode=mode,
            **kwargs,
        )


def encode_class(a):
    pickled_object = cloudpickle.dumps(a, protocol=TILEDB_CLOUD_PROTOCOL)
    return base64.b64encode(pickled_object).decode("ascii")

def get_source_code(a):
    import inspect
    f = open(inspect.getsourcefile(a.__class__), "r")
    return f.read()

def instantiate_object(code, class_name, **kwargs):
    import random
    import string
    import os
    import sys
    temp_file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
    abs_path = os.path.abspath(f"{temp_file_name}.py")
    f = open(abs_path, "w")
    f.write(code)
    f.close()
    sys.path.append(os.path.curdir)
    reader_module = __import__(temp_file_name)
    class_ = getattr(reader_module, class_name)
    os.remove(abs_path)
    return class_(**kwargs)


def create(
    uri: str,
    index_type: str,
    object_reader: ObjectReader,
    embedding: ObjectEmbedding,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    **kwargs,
) -> ObjectIndex:
    with tiledb.scope_ctx(ctx_or_config=config):
        dimensions = embedding.dimensions()
        vector_type = embedding.vector_type()
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
        group.meta["object_reader_source_code"] = get_source_code(object_reader)
        group.meta["object_reader_class_name"] = object_reader.__class__.__name__
        group.meta["object_reader_kwargs"] = encode_class(object_reader.init_kwargs())
        group.meta["embedding_source_code"] = get_source_code(embedding)
        group.meta["embedding_class_name"] = embedding.__class__.__name__
        group.meta["embedding_kwargs"] = encode_class(embedding.init_kwargs())

        embeddings_array_name = storage_formats[index.storage_version]["INPUT_VECTORS_ARRAY_NAME"]
        external_ids_array_name = storage_formats[index.storage_version]["EXTERNAL_IDS_ARRAY_NAME"]
        filters = storage_formats[index.storage_version]["DEFAULT_ATTR_FILTERS"]

        create_embeddings_array(uri, group, dimensions, vector_type, embeddings_array_name, filters)
        create_external_ids_array(uri, group, external_ids_array_name, filters)
        object_metadata_array_uri = object_reader.metadata_array_uri()
        materialize_object_metadata = False
        if object_metadata_array_uri is None and object_reader.metadata_attributes() is not None:
            metadata_array_name = storage_formats[index.storage_version]["OBJECT_METADATA_ARRAY_NAME"]
            object_metadata_array_uri = f"{uri}/{metadata_array_name}"
            external_ids_dim = tiledb.Dim(
                name="external_id",
                domain=(0, np.iinfo(np.dtype("int32")).max),
                dtype=np.dtype(np.int32),
            )
            external_ids_dom = tiledb.Domain(external_ids_dim)
            schema = tiledb.ArraySchema(
                domain=external_ids_dom,
                sparse=True,
                attrs=object_reader.metadata_attributes(),
                # capacity=10000
            )
            tiledb.Array.create(object_metadata_array_uri, schema)
            group.add(object_metadata_array_uri, name=metadata_array_name)
            materialize_object_metadata = True
        object_metadata_external_id_dim = ""
        if object_metadata_array_uri is not None:
            with tiledb.open(object_metadata_array_uri, "r") as object_metadata_array:
                object_metadata_external_id_dim = object_metadata_array.schema.domain.dim(0).name

        group.meta["materialize_object_metadata"] = materialize_object_metadata
        group.meta["object_metadata_array_uri"] = object_metadata_array_uri
        group.meta["object_metadata_external_id_dim"] = object_metadata_external_id_dim
        group.close()
        return ObjectIndex(uri, config, load_embedding=False, **kwargs)


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
