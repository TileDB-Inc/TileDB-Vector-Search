import base64
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional

import json
import numpy as np

import tiledb
import tiledb.vector_search.object_api as object_api
from tiledb.cloud.dag import Mode
from tiledb.vector_search import FlatIndex
from tiledb.vector_search import IVFFlatIndex
from tiledb.vector_search import flat_index
from tiledb.vector_search import ingest
from tiledb.vector_search import ivf_flat_index
from tiledb.vector_search.embeddings import ObjectEmbedding
from tiledb.vector_search.object_readers import ObjectReader
from tiledb.vector_search.storage_formats import STORAGE_VERSION
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.utils import add_to_group

TILEDB_CLOUD_PROTOCOL = 4


class ObjectIndex:
    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        load_embedding: bool = True,
        **kwargs,
    ):
        with tiledb.scope_ctx(ctx_or_config=config):
            self.uri = uri
            self.config = config
            self.timestamp = timestamp
            group = tiledb.Group(uri, "r")
            self.index_type = group.meta["index_type"]
            group.close()
            if self.index_type == "FLAT":
                self.index = FlatIndex(
                    uri=uri, config=config, timestamp=timestamp, **kwargs
                )
            elif self.index_type == "IVF_FLAT":
                self.index = IVFFlatIndex(
                    uri=uri, config=config, timestamp=timestamp, **kwargs
                )

            self.object_reader_source_code = self.index.group.meta[
                "object_reader_source_code"
            ]
            self.object_reader_class_name = self.index.group.meta[
                "object_reader_class_name"
            ]
            self.object_reader_kwargs = json.loads(
                self.index.group.meta["object_reader_kwargs"]
            )
            self.object_reader = instantiate_object(
                code=self.object_reader_source_code,
                class_name=self.object_reader_class_name,
                **self.object_reader_kwargs,
            )
            self.embedding_source_code = self.index.group.meta["embedding_source_code"]
            self.embedding_class_name = self.index.group.meta["embedding_class_name"]
            self.embedding_kwargs = json.loads(
                self.index.group.meta["embedding_kwargs"]
            )
            self.embedding = instantiate_object(
                code=self.embedding_source_code,
                class_name=self.embedding_class_name,
                **self.embedding_kwargs,
            )
            self.embedding_loaded = False
            if load_embedding:
                self.embedding.load()
                self.embedding_loaded = True

            self.materialize_object_metadata = self.index.group.meta[
                "materialize_object_metadata"
            ]
            if "object_metadata_array_uri" in self.index.group.meta:
                self.object_metadata_array_uri = self.index.group.meta[
                    "object_metadata_array_uri"
                ]
                self.object_metadata_external_id_dim = self.index.group.meta[
                    "object_metadata_external_id_dim"
                ]
            else:
                self.object_metadata_array_uri = None
                self.object_metadata_external_id_dim = None

    def query(
        self,
        query_objects: np.ndarray,
        k: int,
        query_metadata: OrderedDict = None,
        metadata_array_cond: str = None,
        metadata_df_filter_fn: str = None,
        return_objects: bool = True,
        return_metadata: bool = True,
        **kwargs,
    ):
        if (
            metadata_array_cond is not None or metadata_df_filter_fn is not None
        ) and self.object_metadata_array_uri is None:
            raise AttributeError(
                "metadata_array_cond and metadata_df_filter_fn can't be applied when there is no metadata array"
            )
        if not self.embedding_loaded:
            self.embedding.load()
            self.embedding_loaded = True
        query_embeddings = self.embedding.embed(
            objects=query_objects, metadata=query_metadata
        )
        fetch_k = k
        if metadata_array_cond is not None or metadata_df_filter_fn is not None:
            fetch_k = min(50 * k, self.index.size)

        distances, object_ids = self.index.query(
            queries=query_embeddings, k=fetch_k, **kwargs
        )
        unique_ids, idx = np.unique(object_ids, return_inverse=True)
        idx = np.reshape(idx, object_ids.shape)

        if metadata_array_cond is not None or metadata_df_filter_fn is not None:
            with tiledb.open(
                self.object_metadata_array_uri,
                mode="r",
                timestamp=self.timestamp,
                config=self.config,
            ) as metadata_array:
                q = metadata_array.query(
                    cond=metadata_array_cond, coords=True, use_arrow=False
                )
                unique_ids_metadata_df = q.df[unique_ids]
                if metadata_df_filter_fn is not None:
                    unique_ids_metadata_df = unique_ids_metadata_df[
                        unique_ids_metadata_df.apply(metadata_df_filter_fn, axis=1)
                    ]
                filtered_unique_ids = unique_ids_metadata_df[
                    self.object_metadata_external_id_dim
                ].to_numpy()
                filtered_distances = np.zeros((query_embeddings.shape[0], k)).astype(
                    object_ids.dtype
                )
                filtered_object_ids = np.zeros((query_embeddings.shape[0], k)).astype(
                    object_ids.dtype
                )
                for query_id in range(query_embeddings.shape[0]):
                    write_id = 0
                    for result_id in range(fetch_k):
                        if object_ids[query_id, result_id] in filtered_unique_ids:
                            filtered_distances[query_id, write_id] = distances[
                                query_id, result_id
                            ]
                            filtered_object_ids[query_id, write_id] = object_ids[
                                query_id, result_id
                            ]
                            write_id += 1
                            if write_id >= k:
                                break

            distances = filtered_distances
            object_ids = filtered_object_ids
            unique_ids, idx = np.unique(object_ids, return_inverse=True)
            idx = np.reshape(idx, object_ids.shape)

        object_metadata = None
        if return_metadata:
            if self.object_metadata_array_uri is not None:
                with tiledb.open(
                    self.object_metadata_array_uri,
                    mode="r",
                    timestamp=self.timestamp,
                    config=self.config,
                ) as metadata_array:
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
        worker_image: str = None,
        extra_worker_modules: Optional[List[str]] = None,
        max_tasks_per_stage: int = -1,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        mode: Mode = Mode.LOCAL,
        config: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ):
        embeddings_array_name = storage_formats[self.index.storage_version][
            "INPUT_VECTORS_ARRAY_NAME"
        ]
        embeddings_uri = f"{self.uri}/{embeddings_array_name}"
        storage_formats[self.index.storage_version]["EXTERNAL_IDS_ARRAY_NAME"]
        metadata_array_uri = None
        if self.materialize_object_metadata:
            metadata_array_uri = self.object_metadata_array_uri
        if config is None:
            config = self.config
        object_api.ingest_embeddings(
            object_index=self,
            embeddings_uri=embeddings_uri,
            metadata_array_uri=metadata_array_uri,
            index_timestamp=index_timestamp,
            max_tasks_per_stage=max_tasks_per_stage,
            workers=workers,
            worker_resources=worker_resources,
            worker_image=worker_image,
            extra_worker_modules=extra_worker_modules,
            verbose=verbose,
            trace_id=trace_id,
            mode=mode,
            config=config,
            namespace=namespace,
            **kwargs,
        )
        self.index = ingest(
            index_type=self.index_type,
            index_uri=self.uri,
            source_uri=embeddings_uri,
            source_type="TILEDB_PARTITIONED_ARRAY",
            external_ids_uri=embeddings_uri,
            external_ids_type="TILEDB_PARTITIONED_ARRAY",
            index_timestamp=index_timestamp,
            storage_version=self.index.storage_version,
            config=config,
            mode=mode,
            **kwargs,
        )


def get_source_code(a):
    import inspect

    f = open(inspect.getsourcefile(a.__class__), "r")
    return f.read()


def instantiate_object(code, class_name, **kwargs):
    import importlib.util
    import os
    import random
    import string
    import sys

    temp_file_name = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=16)
    )
    abs_path = os.path.abspath(f"{temp_file_name}.py")
    f = open(abs_path, "w")
    f.write(code)
    f.close()
    spec = importlib.util.spec_from_file_location(temp_file_name, abs_path)
    reader_module = importlib.util.module_from_spec(spec)
    sys.modules[temp_file_name] = reader_module
    spec.loader.exec_module(reader_module)
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
                storage_version=storage_version,
            )
        elif index_type == "IVF_FLAT":
            index = ivf_flat_index.create(
                uri=uri,
                dimensions=dimensions,
                vector_type=vector_type,
                group_exists=False,
                config=config,
                storage_version=storage_version,
            )

        group = tiledb.Group(uri, "w")
        group.meta["object_reader_source_code"] = get_source_code(object_reader)
        group.meta["object_reader_class_name"] = object_reader.__class__.__name__
        group.meta["object_reader_kwargs"] = json.dumps(object_reader.init_kwargs())
        group.meta["embedding_source_code"] = get_source_code(embedding)
        group.meta["embedding_class_name"] = embedding.__class__.__name__
        group.meta["embedding_kwargs"] = json.dumps(embedding.init_kwargs())

        embeddings_array_name = storage_formats[index.storage_version][
            "INPUT_VECTORS_ARRAY_NAME"
        ]
        filters = storage_formats[index.storage_version]["DEFAULT_ATTR_FILTERS"]

        create_embeddings_partitioned_array(
            uri, embeddings_array_name, group, vector_type, filters, config
        )
        object_metadata_array_uri = object_reader.metadata_array_uri()
        materialize_object_metadata = False
        if (
            object_metadata_array_uri is None
            and object_reader.metadata_attributes() is not None
        ):
            metadata_array_name = storage_formats[index.storage_version][
                "OBJECT_METADATA_ARRAY_NAME"
            ]
            object_metadata_array_uri = f"{uri}/{metadata_array_name}"
            external_ids_dim = tiledb.Dim(
                name="external_id",
                domain=(0, np.iinfo(np.dtype("uint64")).max - 10000),
                tile=10000,
                dtype=np.dtype(np.uint64),
            )
            external_ids_dom = tiledb.Domain(external_ids_dim)
            schema = tiledb.ArraySchema(
                domain=external_ids_dom,
                sparse=True,
                attrs=object_reader.metadata_attributes(),
            )
            tiledb.Array.create(object_metadata_array_uri, schema)
            add_to_group(group, object_metadata_array_uri, name=metadata_array_name)
            materialize_object_metadata = True
        object_metadata_external_id_dim = ""
        if object_metadata_array_uri is not None:
            with tiledb.open(object_metadata_array_uri, "r") as object_metadata_array:
                object_metadata_external_id_dim = (
                    object_metadata_array.schema.domain.dim(0).name
                )
            group.meta["object_metadata_array_uri"] = object_metadata_array_uri
            group.meta[
                "object_metadata_external_id_dim"
            ] = object_metadata_external_id_dim

        group.meta["materialize_object_metadata"] = materialize_object_metadata
        group.close()
        return ObjectIndex(uri, config, load_embedding=False, **kwargs)


def create_embeddings_partitioned_array(
    group_uri: str,
    array_name: str,
    group: tiledb.Group,
    vector_type: np.dtype,
    filters,
    config: Optional[Mapping[str, Any]] = None,
):
    with tiledb.scope_ctx(ctx_or_config=config):
        embeddings_array_uri = f"{group_uri}/{array_name}"
        if tiledb.array_exists(embeddings_array_uri):
            raise ValueError(f"Array exists {embeddings_array_uri}")
        partition_id_dim = tiledb.Dim(
            name="partition_id",
            domain=(0, np.iinfo(np.dtype("uint32")).max - 1),
            tile=1,
            dtype=np.dtype(np.uint32),
        )
        domain = tiledb.Domain(partition_id_dim)
        attrs = [
            tiledb.Attr(name="vectors", dtype=vector_type, var=True, filters=filters),
            tiledb.Attr(
                name="vectors_shape", dtype=np.uint32, var=True, filters=filters
            ),
            tiledb.Attr(
                name="external_ids",
                dtype=np.dtype(np.uint64),
                var=True,
                filters=filters,
            ),
        ]
        embeddings_array_schema = tiledb.ArraySchema(
            domain=domain,
            sparse=False,
            attrs=attrs,
        )
        tiledb.Array.create(embeddings_array_uri, embeddings_array_schema)
        add_to_group(group, embeddings_array_uri, name=array_name)
