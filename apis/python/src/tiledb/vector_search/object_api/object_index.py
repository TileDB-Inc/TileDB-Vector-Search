import json
import random
import string
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

import tiledb
import tiledb.vector_search.object_api as object_api
from tiledb.cloud.dag import Mode
from tiledb.vector_search import FlatIndex
from tiledb.vector_search import IVFFlatIndex
from tiledb.vector_search import IVFPQIndex
from tiledb.vector_search import VamanaIndex
from tiledb.vector_search import flat_index
from tiledb.vector_search import ivf_flat_index
from tiledb.vector_search import ivf_pq_index
from tiledb.vector_search import vamana_index
from tiledb.vector_search.embeddings import ObjectEmbedding
from tiledb.vector_search.object_readers import ObjectReader
from tiledb.vector_search.storage_formats import STORAGE_VERSION
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.utils import add_to_group

TILEDB_CLOUD_PROTOCOL = 4


class ObjectIndex:
    """
    An ObjectIndex represents a TileDB Vector Search index that is associated with a
    user-defined object reader and embedding function. This allows users to easily
    create and query TileDB Vector Search indexes that are backed by arbitrary data.

    For example, an ObjectIndex can be used to create a TileDB Vector Search index
    that is backed by a collection of images. The object reader would be responsible
    for loading the images from disk, and the embedding function would be responsible
    for generating embeddings for the images.

    Once the ObjectIndex is created, it can be queried using the `query()` method.
    The `query()` method takes a list of query objects and returns a list of the
    nearest neighbors for each query object.

    The ObjectIndex class also provides methods for updating the index
    (`update_index()`) and updating the object reader (`update_object_reader()`).

    Parameters
    ----------
    uri: str
        The URI of the index.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    timestamp: Optional[int]
        Timestamp to open the index at.
    load_embedding: bool
        Whether to load the embedding function into memory.
    load_metadata_in_memory: bool
        Whether to load the metadata array into memory.
    environment_variables: Dict
        Environment variables to set for the object reader and embedding function.
    **kwargs
        Keyword arguments to pass to the index constructor.
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        load_embedding: bool = True,
        load_metadata_in_memory: bool = True,
        environment_variables: Dict = {},
        **kwargs,
    ):
        import os

        for var, val in environment_variables.items():
            os.environ[var] = val
        with tiledb.scope_ctx(ctx_or_config=config):
            self.uri = uri
            self.config = config
            self.timestamp = timestamp
            self.load_metadata_in_memory = load_metadata_in_memory
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
            elif self.index_type == "VAMANA":
                self.index = VamanaIndex(
                    uri=uri, config=config, timestamp=timestamp, **kwargs
                )
            elif self.index_type == "IVF_PQ":
                self.index = IVFPQIndex(
                    uri=uri, config=config, timestamp=timestamp, **kwargs
                )
            else:
                raise ValueError(f"Unsupported index type {self.index_type}")

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

            if self.object_metadata_array_uri is not None:
                self.metadata_array = tiledb.open(
                    self.object_metadata_array_uri,
                    mode="r",
                    timestamp=self.timestamp,
                    config=self.config,
                )
                if self.load_metadata_in_memory:
                    self.metadata_df = self.metadata_array.df[:]

    def query(
        self,
        query_objects: np.ndarray,
        k: int,
        query_metadata: Optional[OrderedDict] = None,
        metadata_array_cond: Optional[str] = None,
        metadata_df_filter_fn: Optional[str] = None,
        return_objects: bool = True,
        return_metadata: bool = True,
        **kwargs,
    ):
        """
        Queries the index and returns the nearest neighbors for each query object.

        The query objects can be any type of object that is supported by the object
        reader. For example, if the object reader is configured to read images, then
        the query objects should be images.

        The `k` parameter specifies the number of nearest neighbors to return for
        each query object.

        The `query_metadata` parameter can be used to pass metadata for the query
        objects. This metadata will be passed to the embedding function, which can
        use it to generate embeddings for the query objects.

        The `metadata_array_cond` parameter can be used to filter the results of the
        query based on the metadata that is stored in the metadata array. This
        parameter should be a string that contains a valid TileDB query condition.
        For example, the following query condition could be used to filter the
        results to only include objects that have a color attribute that is equal
        to "red":

        ```python
        metadata_array_cond="color='red'"
        ```

        The `metadata_df_filter_fn` parameter can be used to filter the results of the
        query based on the metadata that is stored in the metadata array. This
        parameter should be a function that takes a pandas DataFrame as input and
        returns a pandas DataFrame as output. The input DataFrame will contain the
        metadata for all of the objects that match the query, and the output DataFrame
        should contain the metadata for the objects that should be returned to the
        user.

        The `return_objects` parameter specifies whether to return the objects
        themselves, or just the object IDs. If this parameter is set to `True`, then
        the `query()` method will also return the objects instead of the object IDs.

        The `return_metadata` parameter specifies whether to return the metadata for
        the objects. If this parameter is set to `True`, then the `query()` method
        will also return the object metadata along with the distances and object IDs.

        Parameters
        ----------
        query_objects: np.ndarray
            The query objects.
        k: int
            The number of nearest neighbors to return for each query object.
        query_metadata: Optional[OrderedDict]
            Metadata for the query objects.
        metadata_array_cond: Optional[str]
            A TileDB query condition that can be used to filter the results of the
            query based on the metadata that is stored in the metadata array.
        metadata_df_filter_fn: Optional[str]
            A function that can be used to filter the results of the query based on
            the metadata that is stored in the metadata array.
        return_objects: bool
            Whether to return the objects themselves, or just the object IDs.
        return_metadata: bool
            Whether to return the metadata for the objects.
        **kwargs
            Keyword arguments to pass to the index query method.

        Returns
        -------
        Union[
            Tuple[np.ndarray, OrderedDict, Dict],
            Tuple[np.ndarray, OrderedDict],
            Tuple[np.ndarray, np.ndarray, Dict],
            Tuple[np.ndarray, np.ndarray],
        ]
            A tuple containing the distances, objects or object IDs, and optionally
            the object metadata.
        """
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
            if self.load_metadata_in_memory:
                if metadata_array_cond is not None:
                    raise AttributeError(
                        "metadata_array_cond is not supported with load_metadata_in_memory. Please use metadata_df_filter_fn."
                    )
                unique_ids_metadata_df = self.metadata_df[
                    self.metadata_df[self.object_metadata_external_id_dim].isin(
                        unique_ids
                    )
                ]
            else:
                q = self.metadata_array.query(
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
                if self.load_metadata_in_memory:
                    unique_metadata = self.metadata_df[
                        self.metadata_df[self.object_metadata_external_id_dim].isin(
                            unique_ids
                        )
                    ]
                    object_metadata = {}
                    for attr in unique_metadata.keys():
                        object_metadata[attr] = unique_metadata[attr].to_numpy()[idx]
                else:
                    unique_metadata = self.metadata_array.multi_index[unique_ids]
                    object_metadata = {}
                    for attr in unique_metadata.keys():
                        unique_metadata[attr][idx]
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

    def update_object_reader(
        self,
        object_reader: ObjectReader,
        config: Optional[Mapping[str, Any]] = None,
    ):
        """
        Updates the object reader for the index.

        This method can be used to update the object reader for the index. This is
        useful if the object reader needs to be updated to read objects from a
        different location, or if the object reader needs to be updated to read
        objects in a different format.

        Parameters
        ----------
        object_reader: ObjectReader
            The new object reader.
        config: Optional[Mapping[str, Any]]
            TileDB config dictionary.
        """
        with tiledb.scope_ctx(ctx_or_config=config):
            self.object_reader = object_reader
            self.object_reader_source_code = get_source_code(object_reader)
            self.object_reader_class_name = object_reader.__class__.__name__
            self.object_reader_kwargs = json.dumps(object_reader.init_kwargs())
            group = tiledb.Group(self.uri, "w")
            group.meta["object_reader_source_code"] = self.object_reader_source_code
            group.meta["object_reader_class_name"] = self.object_reader_class_name
            group.meta["object_reader_kwargs"] = self.object_reader_kwargs
            group.close()

    def create_embeddings_partitioned_array(self) -> (str, str):
        """
        Creates a partitioned array for storing embeddings.

        This method is used internally by the `update_index()` method to create a
        partitioned array for storing embeddings.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the name of the temporary directory and the URI of the
            embeddings array.
        """
        temp_dir_name = (
            storage_formats[self.index.storage_version]["PARTIAL_WRITE_ARRAY_DIR"]
            + "_"
            + "".join(random.choices(string.ascii_letters, k=10))
        )
        temp_dir_uri = f"{self.uri}/{temp_dir_name}"
        try:
            tiledb.group_create(temp_dir_uri)
            group = tiledb.Group(self.uri, "w")
            add_to_group(group, temp_dir_uri, temp_dir_name)
            group.close()
        except tiledb.TileDBError as err:
            message = str(err)
            if "already exists" not in message:
                raise err

        embeddings_array_name = storage_formats[self.index.storage_version][
            "INPUT_VECTORS_ARRAY_NAME"
        ]
        filters = storage_formats[self.index.storage_version]["DEFAULT_ATTR_FILTERS"]
        embeddings_array_uri = f"{temp_dir_uri}/{embeddings_array_name}"
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
            tiledb.Attr(
                name="vectors", dtype=self.index.dtype, var=True, filters=filters
            ),
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
        temp_dir_group = tiledb.Group(temp_dir_uri, "w")
        add_to_group(temp_dir_group, embeddings_array_uri, name=embeddings_array_name)
        temp_dir_group.close()
        return temp_dir_name, embeddings_array_uri

    def update_index(
        self,
        index_timestamp: int = None,
        workers: int = -1,
        worker_resources: Dict = None,
        worker_image: str = None,
        extra_worker_modules: Optional[List[str]] = None,
        driver_resources: Dict = None,
        driver_image: str = None,
        extra_driver_modules: Optional[List[str]] = None,
        worker_access_credentials_name: str = None,
        max_tasks_per_stage: int = -1,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        embeddings_generation_mode: Mode = Mode.LOCAL,
        embeddings_generation_driver_mode: Mode = Mode.LOCAL,
        vector_indexing_mode: Mode = Mode.LOCAL,
        config: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        environment_variables: Dict = {},
        **kwargs,
    ):
        """Updates the index with new data.

        This method can be used to update the index with new data. This is useful
        if the data that the index is built on has changed.

        Update uses the `ingest_embeddings_with_driver` function to add embeddings
        into a TileDB vector search index.

        This function orchestrates the embedding ingestion process by creating and executing
        a TileDB Cloud DAG (Directed Acyclic Graph). The DAG consists of two main stages:

        1. **Embeddings Generation:** This stage is responsible for computing embeddings
        for the objects to be indexed. It can be executed in one of three modes:

        - **LOCAL:** Embeddings are generated locally within the current process.
        - **REALTIME:** Embeddings are generated using a TileDB Cloud REALTIME TaskGraph.
        - **BATCH:** Embeddings are generated using a TileDB Cloud BATCH TaskGraph.

        2. **Vector Indexing:** This stage is responsible for ingesting the generated
        embeddings into the TileDB vector search index. It can be executed in one of
        three modes:

        - **LOCAL:** Embeddings are ingested locally within the current process.
        - **REALTIME:** Embeddings are ingested using a TileDB Cloud REALTIME TaskGraph.
        - **BATCH:** Embeddings are ingested using a TileDB Cloud BATCH TaskGraph.

        The `ingest_embeddings_with_driver` function provides flexibility in configuring
        the execution environment for both stages as well as can run the full execution within a
        driver UDF. Users can specify the number of workers, resources, Docker images, and extra
        modules for both the driver and worker nodes.

        The `update_index()` method takes the following parameters:

        Parameters
        ----------
        index_timestamp: int
            Timestamp to use for the update to take place at.
        workers: int
            The number of workers to use for the update. If this parameter is not
            specified, then the default number of workers will be used.
        worker_resources: Dict
            The resources to use for each worker.
        worker_image: str
            The Docker image to use for each worker.
        extra_worker_modules: Optional[List[str]]
            Extra modules to install on the worker nodes.
        driver_resources: Dict
            The resources to use for the driver.
        driver_image: str
            The Docker image to use for the driver.
        extra_driver_modules: Optional[List[str]]
            Extra modules to install on the driver node.
        worker_access_credentials_name: str
            The name of the TileDB Cloud access credentials to use for the
            workers.
        max_tasks_per_stage: int
            The maximum number of tasks to run per stage.
        verbose: bool
            Whether to print verbose output.
        trace_id: Optional[str]
            The trace ID to use for the update.
        embeddings_generation_mode: Mode
            The mode to use for generating embeddings.
        embeddings_generation_driver_mode: Mode
            The mode to use for the driver of the embeddings generation task.
        vector_indexing_mode: Mode
            The mode to use for indexing the vectors.
        config: Optional[Mapping[str, Any]]
            TileDB config dictionary.
        namespace: Optional[str]
            The TileDB Cloud namespace to use for the update. If this parameter
            is not specified, then the default namespace will be used.
        environment_variables: Dict
            Environment variables to set for the object reader and embedding function.
        **kwargs
            Keyword arguments to pass to the ingestion function.
        """
        with tiledb.scope_ctx(ctx_or_config=config):
            use_updates_array = True
            embeddings_array_uri = None
            if self.index.size == 0:
                (
                    temp_dir_name,
                    embeddings_array_uri,
                ) = self.create_embeddings_partitioned_array()
                use_updates_array = False

            storage_formats[self.index.storage_version]["EXTERNAL_IDS_ARRAY_NAME"]
            metadata_array_uri = None
            if self.materialize_object_metadata:
                metadata_array_uri = self.object_metadata_array_uri
            if config is None:
                config = self.config

            object_api.ingest_embeddings_with_driver(
                object_index_uri=self.uri,
                use_updates_array=use_updates_array,
                embeddings_array_uri=embeddings_array_uri,
                metadata_array_uri=metadata_array_uri,
                index_timestamp=index_timestamp,
                max_tasks_per_stage=max_tasks_per_stage,
                workers=workers,
                worker_resources=worker_resources,
                worker_image=worker_image,
                extra_worker_modules=extra_worker_modules,
                driver_resources=driver_resources,
                driver_image=driver_image,
                extra_driver_modules=extra_driver_modules,
                worker_access_credentials_name=worker_access_credentials_name,
                verbose=verbose,
                trace_id=trace_id,
                embeddings_generation_driver_mode=embeddings_generation_driver_mode,
                embeddings_generation_mode=embeddings_generation_mode,
                vector_indexing_mode=vector_indexing_mode,
                config=config,
                namespace=namespace,
                environment_variables=environment_variables,
                **kwargs,
            )

            if not use_updates_array:
                with tiledb.Group(self.uri, "w") as group:
                    group.remove(temp_dir_name)
                temp_dir_uri = f"{self.uri}/{temp_dir_name}"
                with tiledb.Group(temp_dir_uri, "m") as temp_dir_group:
                    temp_dir_group.delete(recursive=True)


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
    """Creates a new ObjectIndex.

    Parameters
    ----------
    uri: str
        The URI of the index.
    index_type: str
        The type of index to create. Can be one of "FLAT", "IVF_FLAT", "VAMANA", or "IVF_PQ".
    object_reader: ObjectReader
        The object reader to use for the index.
    embedding: ObjectEmbedding
        The embedding function to use for the index.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    storage_version: str
        The storage version to use for the index.
    **kwargs
        Keyword arguments to pass to the index constructor.

    Returns
    -------
    ObjectIndex
        The newly created ObjectIndex.
    """
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
        elif index_type == "VAMANA":
            index = vamana_index.create(
                uri=uri,
                dimensions=dimensions,
                vector_type=vector_type,
                config=config,
                storage_version=storage_version,
            )
        elif index_type == "IVF_PQ":
            if "num_subspaces" not in kwargs:
                raise ValueError(
                    "num_subspaces must be provided when creating an IVF_PQ index"
                )
            num_subspaces = kwargs["num_subspaces"]
            partitions = kwargs.get("partitions", None)
            index = ivf_pq_index.create(
                uri=uri,
                dimensions=dimensions,
                vector_type=vector_type,
                config=config,
                storage_version=storage_version,
                partitions=partitions,
                num_subspaces=num_subspaces,
            )
        else:
            raise ValueError(f"Unsupported index type {index_type}")

        group = tiledb.Group(uri, "w")
        group.meta["object_reader_source_code"] = get_source_code(object_reader)
        group.meta["object_reader_class_name"] = object_reader.__class__.__name__
        group.meta["object_reader_kwargs"] = json.dumps(object_reader.init_kwargs())
        group.meta["embedding_source_code"] = get_source_code(embedding)
        group.meta["embedding_class_name"] = embedding.__class__.__name__
        group.meta["embedding_kwargs"] = json.dumps(embedding.init_kwargs())
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
        return ObjectIndex(
            uri, config, load_embedding=False, load_metadata_in_memory=False, **kwargs
        )
