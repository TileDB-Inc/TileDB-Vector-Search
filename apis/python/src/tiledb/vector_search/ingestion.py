"""
Vector Search ingestion Utilities

This contains the ingestion implementation for different TileDB Vector Search algorithms.

It enables:

- Local ingestion:
  - Multi-threaded execution that can leverage all the available local computing resources.
- Distributed ingestion:
  - Distributed ingestion execution with multiple workers in TileDB Cloud. This can be used
  to ingest large datasets and speedup ingestion latency.
"""

import enum
from functools import partial
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from tiledb.cloud.dag import Mode
from tiledb.vector_search._tiledbvspy import *
from tiledb.vector_search.storage_formats import STORAGE_VERSION
from tiledb.vector_search.storage_formats import validate_storage_version
from tiledb.vector_search.utils import add_to_group
from tiledb.vector_search.utils import is_type_erased_index
from tiledb.vector_search.utils import to_temporal_policy


class TrainingSamplingPolicy(enum.Enum):
    FIRST_N = 1
    RANDOM = 2

    def __str__(self):
        return self.name.replace("_", " ").title()


def ingest(
    index_type: str,
    index_uri: str,
    *,
    input_vectors: np.ndarray = None,
    source_uri: str = None,
    source_type: str = None,
    external_ids: np.array = None,
    external_ids_uri: str = "",
    external_ids_type: str = None,
    updates_uri: str = None,
    index_timestamp: int = None,
    config: Optional[Mapping[str, Any]] = None,
    namespace: Optional[str] = None,
    size: int = -1,
    partitions: int = -1,
    num_subspaces: int = -1,
    l_build: int = -1,
    r_max_degree: int = -1,
    training_sampling_policy: TrainingSamplingPolicy = TrainingSamplingPolicy.FIRST_N,
    copy_centroids_uri: str = None,
    training_sample_size: int = -1,
    training_input_vectors: np.ndarray = None,
    training_source_uri: str = None,
    training_source_type: str = None,
    workers: int = -1,
    input_vectors_per_work_item: int = -1,
    max_tasks_per_stage: int = -1,
    input_vectors_per_work_item_during_sampling: int = -1,
    max_sampling_tasks: int = -1,
    storage_version: str = STORAGE_VERSION,
    verbose: bool = False,
    trace_id: Optional[str] = None,
    use_sklearn: bool = True,
    mode: Mode = Mode.LOCAL,
    acn: Optional[str] = None,
    ingest_resources: Optional[Mapping[str, Any]] = None,
    consolidate_partition_resources: Optional[Mapping[str, Any]] = None,
    copy_centroids_resources: Optional[Mapping[str, Any]] = None,
    random_sample_resources: Optional[Mapping[str, Any]] = None,
    kmeans_resources: Optional[Mapping[str, Any]] = None,
    compute_new_centroids_resources: Optional[Mapping[str, Any]] = None,
    assign_points_and_partial_new_centroids_resources: Optional[
        Mapping[str, Any]
    ] = None,
    write_centroids_resources: Optional[Mapping[str, Any]] = None,
    partial_index_resources: Optional[Mapping[str, Any]] = None,
    **kwargs,
):
    """
    Ingest vectors into TileDB.

    Parameters
    ----------
    index_type: str
        Type of vector index (FLAT, IVF_FLAT, IVF_PQ, VAMANA).
    index_uri: str
        Vector index URI (stored as TileDB group).
    input_vectors: np.ndarray
        Input vectors, if this is provided it takes precedence over `source_uri` and `source_type`.
    source_uri: str
        Vectors source URI.
    source_type: str
        Type of the source vectors. If left empty it is auto-detected.
    external_ids: np.array
        Input vector `external_ids`, if this is provided it takes precedence over `external_ids_uri` and `external_ids_type`.
    external_ids_uri: str
        Source URI for `external_ids`.
    external_ids_type: str
        File type of external_ids_uri. If left empty it is auto-detected.
    updates_uri: str
        Updates array URI. Used for consolidation of updates.
    index_timestamp: int
        Timestamp to use for writing and reading data. By default it uses the current unix ms timestamp.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    namespace: str
        TileDB-Cloud namespace to use for Cloud execution.
    size: int
        Number of input vectors, if not provided use the full size of the input dataset.
        If provided, we filter the first vectors from the input source.
    partitions: int
        For IVF indexes, the number of partitions to load the data with, if not provided, is auto-configured based on the dataset size.
    num_subspaces: int
        For PQ encoded indexes, the number of subspaces to use in the PQ encoding. We will divide the dimensions into
        num_subspaces parts, and PQ encode each part separately. This means dimensions must
        be divisible by num_subspaces.
    l_build: int
        For Vamana indexes, the number of neighbors considered for each node during construction of the graph. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. l_build should be >= r_max_degree unless you need to build indices quickly and can compromise on quality.
        Typically between 75 and 200. If not provided, use the default value of 100.
    r_max_degree: int
        For Vamana indexes, the maximum degree for each node in the final graph. Larger values will result in larger indices and longer indexing times, but better search quality.
        Typically between 60 and 150. If not provided, use the default value of 64.
    copy_centroids_uri: str
        TileDB array URI to copy centroids from, if not provided, centroids are build running `k-means`.
    training_sample_size: int
        Sample size to use for computing `k-means`. If not provided, is auto-configured based on the dataset sizes.
        Should not be provided if training_source_uri is provided.
    training_input_vectors: np.ndarray
        Training input vectors, if this is provided it takes precedence over `training_source_uri` and `training_source_type`.
        Should not be provided if `training_sample_size` or `training_source_uri` are provided.
    training_source_uri: str
        The source URI to use for training centroids when building a `IVF_FLAT` vector index.
        If not provided, the first `training_sample_size` vectors from `source_uri` are used.
        Should not be provided if training_sample_size or training_input_vectors is provided.
    training_source_type: str
        Type of the training source data in `training_source_uri`.
        If left empty, is auto-detected. Should only be provided when `training_source_uri` is provided.
    workers: int
        Number of distributed workers to use for vector ingestion.
        If not provided, is auto-configured based on the dataset size.
    input_vectors_per_work_item: int
        Number of vectors per ingestion work item.
        If not provided, is auto-configured.
    max_tasks_per_stage: int
        Max number of tasks per execution stage of ingestion.
        If not provided, is auto-configured.
    input_vectors_per_work_item_during_sampling: int
        Number of vectors per sample ingestion work item.
        iIf not provided, is auto-configured.
        Only valid with `training_sampling_policy=TrainingSamplingPolicy.RANDOM`.
    max_sampling_tasks: int
        Max number of tasks per execution stage of sampling.
        If not provided, is auto-configured
        Only valid with `training_sampling_policy=TrainingSamplingPolicy.RANDOM`.
    storage_version: str
        Vector index storage format version. If not provided, defaults to the latest version.
    verbose: bool
        Enables verbose logging.
    trace_id: Optional[str]
        trace ID for logging.
    use_sklearn: bool
        Whether to use scikit-learn's implementation of k-means clustering instead of
        tiledb.vector_search's.
    mode: Mode
        Execution mode, defaults to `LOCAL` use `BATCH` for distributed execution.
    acn: Optional[str]
        Access credential name to be used when running in BATCH mode for object store access
    ingest_resources: Optional[Mapping[str, Any]]
        Resources to request when performing vector ingestion, only applies to BATCH mode
    consolidate_partition_resources: Optional[Mapping[str, Any]]
        Resources to request when performing consolidation of a partition, only applies to BATCH mode
    copy_centroids_resources: Optional[Mapping[str, Any]]
        Resources to request when performing copy of centroids from input array to output array, only applies to BATCH mode
    random_sample_resources: Optional[Mapping[str, Any]]
        Resources to request when performing random sample selection, only applies to BATCH mode
    kmeans_resources: Optional[Mapping[str, Any]]
        Resources to request when performing kmeans task, only applies to BATCH mode
    compute_new_centroids_resources: Optional[Mapping[str, Any]]
        Resources to request when performing centroid computation, only applies to BATCH mode
    assign_points_and_partial_new_centroids_resources: Optional[Mapping[str, Any]]
        Resources to request when performing the computation of partial centroids, only applies to BATCH mode
    write_centroids_resources: Optional[Mapping[str, Any]]
        Resources to request when performing the write of centroids, only applies to BATCH mode
    partial_index_resources: Optional[Mapping[str, Any]]
        Resources to request when performing the computation of partial indexing, only applies to BATCH mode
    """
    import enum
    import json
    import logging
    import math
    import multiprocessing
    import os
    import random
    import string
    import time
    from typing import Any, Mapping

    import numpy as np

    import tiledb
    from tiledb.cloud import dag
    from tiledb.cloud.rest_api import models
    from tiledb.cloud.utilities import get_logger
    from tiledb.cloud.utilities import set_aws_context
    from tiledb.vector_search import flat_index
    from tiledb.vector_search import ivf_flat_index
    from tiledb.vector_search import ivf_pq_index
    from tiledb.vector_search import vamana_index
    from tiledb.vector_search.storage_formats import storage_formats

    validate_storage_version(storage_version)

    if source_type and not source_uri:
        raise ValueError("source_type should not be provided without source_uri")
    if source_uri and input_vectors:
        raise ValueError("source_uri should not be provided alongside input_vectors")
    if source_type and input_vectors:
        raise ValueError("source_type should not be provided alongside input_vectors")

    if training_source_uri and training_sample_size != -1:
        raise ValueError(
            "training_source_uri and training_sample_size should not both be provided"
        )
    if training_source_uri and training_input_vectors is not None:
        raise ValueError(
            "training_source_uri and training_input_vectors should not both be provided"
        )

    if training_input_vectors is not None and training_sample_size != -1:
        raise ValueError(
            "training_input_vectors and training_sample_size should not both be provided"
        )
    if training_input_vectors is not None and training_source_type:
        raise ValueError(
            "training_input_vectors and training_source_type should not both be provided"
        )

    if training_source_type and not training_source_uri:
        raise ValueError(
            "training_source_type should not be provided without training_source_uri"
        )

    if training_sample_size < -1:
        raise ValueError(
            "training_sample_size should either be positive or -1 (to auto-configure based on the dataset sizes)"
        )

    if copy_centroids_uri is not None and training_sample_size != -1:
        raise ValueError(
            "training_sample_size should not be provided alongside copy_centroids_uri"
        )
    if copy_centroids_uri is not None and partitions == -1:
        raise ValueError(
            "partitions should be provided if copy_centroids_uri is provided (set partitions to the number of centroids in copy_centroids_uri)"
        )

    if index_type != "IVF_FLAT" and training_sample_size != -1:
        raise ValueError(
            "training_sample_size should only be provided with index_type IVF_FLAT"
        )
    for variable in [
        "copy_centroids_uri",
        "training_input_vectors",
        "training_source_uri",
        "training_source_type",
    ]:
        if index_type != "IVF_FLAT" and locals().get(variable) is not None:
            raise ValueError(
                f"{variable} should only be provided with index_type IVF_FLAT"
            )

    for variable in [
        "copy_centroids_uri",
        "training_input_vectors",
        "training_source_uri",
        "training_source_type",
    ]:
        if (
            training_sampling_policy != TrainingSamplingPolicy.FIRST_N
            and locals().get(variable) is not None
        ):
            raise ValueError(
                f"{variable} should not provided alonside training_sampling_policy"
            )

    # use index_group_uri for internal clarity
    index_group_uri = index_uri

    CENTROIDS_ARRAY_NAME = storage_formats[storage_version]["CENTROIDS_ARRAY_NAME"]
    INDEX_ARRAY_NAME = storage_formats[storage_version]["INDEX_ARRAY_NAME"]
    IDS_ARRAY_NAME = storage_formats[storage_version]["IDS_ARRAY_NAME"]
    PARTS_ARRAY_NAME = storage_formats[storage_version]["PARTS_ARRAY_NAME"]
    INPUT_VECTORS_ARRAY_NAME = storage_formats[storage_version][
        "INPUT_VECTORS_ARRAY_NAME"
    ]
    TRAINING_INPUT_VECTORS_ARRAY_NAME = storage_formats[storage_version][
        "TRAINING_INPUT_VECTORS_ARRAY_NAME"
    ]
    EXTERNAL_IDS_ARRAY_NAME = storage_formats[storage_version][
        "EXTERNAL_IDS_ARRAY_NAME"
    ]
    PARTIAL_WRITE_ARRAY_DIR = (
        storage_formats[storage_version]["PARTIAL_WRITE_ARRAY_DIR"]
        + "_"
        + "".join(random.choices(string.ascii_letters, k=10))
    )
    DEFAULT_ATTR_FILTERS = storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"]
    DEFAULT_PARTITION_BYTE_SIZE = 2560000000  # 2.5GB
    VECTORS_PER_SAMPLE_WORK_ITEM = 1000000
    MAX_TASKS_PER_STAGE = 100
    CENTRALISED_KMEANS_MAX_SAMPLE_SIZE = 1000000
    DEFAULT_KMEANS_BYTES_PER_SAMPLE = 128000000  # ~ 128MB
    DEFAULT_IMG_NAME = "3.9-vectorsearch"
    MAX_INT32 = 2**31 - 1

    class SourceType(enum.Enum):
        """SourceType of input vectors"""

        TILEDB_ARRAY = enum.auto()
        U8BIN = enum.auto()

        def __str__(self):
            return self.name.replace("_", " ").title()

    def setup(
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
    ) -> logging.Logger:
        """
        Set the default TileDB context, OS environment variables for AWS,
        and return a logger instance.

        :param config: config dictionary, defaults to None
        :param verbose: verbose logging, defaults to False
        :return: logger instance
        """

        try:
            tiledb.default_ctx(config)
        except tiledb.TileDBError:
            # Ignore error if the default context was already set
            pass

        set_aws_context(config)

        level = logging.DEBUG if verbose else logging.NOTSET
        logger = get_logger(level)

        logger.debug(
            "tiledb.cloud=%s, tiledb=%s, libtiledb=%s",
            tiledb.cloud.version.version,
            tiledb.version(),
            tiledb.libtiledb.version(),
        )

        return logger

    def autodetect_source_type(source_uri: str) -> str:
        if source_uri.endswith(".u8bin"):
            return "U8BIN"
        elif source_uri.endswith(".f32bin"):
            return "F32BIN"
        elif source_uri.endswith(".fvecs"):
            return "FVEC"
        elif source_uri.endswith(".ivecs"):
            return "IVEC"
        elif source_uri.endswith(".bvecs"):
            return "BVEC"
        else:
            return "TILEDB_ARRAY"

    def read_source_metadata(
        source_uri: str, source_type: str = None
    ) -> Tuple[int, int, np.dtype]:
        if source_type == "TILEDB_ARRAY":
            schema = tiledb.ArraySchema.load(source_uri)
            size = schema.domain.dim(1).domain[1] + 1
            dimensions = schema.domain.dim(0).domain[1] + 1
            return size, dimensions, schema.attr(0).dtype
        if source_type == "TILEDB_SPARSE_ARRAY":
            schema = tiledb.ArraySchema.load(source_uri)
            size = schema.domain.dim(0).domain[1] + 1
            dimensions = schema.domain.dim(1).domain[1] + 1
            return size, dimensions, schema.attr(0).dtype
        if source_type == "TILEDB_PARTITIONED_ARRAY":
            with tiledb.open(source_uri, "r", config=config) as source_array:
                q = source_array.query(attrs=("vectors_shape",), coords=True)
                nonempty_object_array_domain = source_array.nonempty_domain()
                partition_shapes = q[
                    nonempty_object_array_domain[0][0] : nonempty_object_array_domain[
                        0
                    ][1]
                    + 1
                ]["vectors_shape"]
                size = 0
                for partition_shape in partition_shapes:
                    size += partition_shape[0]
                    dimensions = partition_shape[1]
                return size, dimensions, source_array.schema.attr("vectors").dtype
        elif source_type == "U8BIN":
            vfs = tiledb.VFS()
            with vfs.open(source_uri, "rb") as f:
                size = int.from_bytes(f.read(4), "little")
                dimensions = int.from_bytes(f.read(4), "little")
                return size, dimensions, np.uint8
        elif source_type == "F32BIN":
            vfs = tiledb.VFS()
            with vfs.open(source_uri, "rb") as f:
                size = int.from_bytes(f.read(4), "little")
                dimensions = int.from_bytes(f.read(4), "little")
                return size, dimensions, np.float32
        elif source_type == "FVEC":
            vfs = tiledb.VFS()
            with vfs.open(source_uri, "rb") as f:
                dimensions = int.from_bytes(f.read(4), "little")
                vector_size = 4 + dimensions * 4
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                size = int(file_size / vector_size)
                return size, dimensions, np.float32
        elif source_type == "IVEC":
            vfs = tiledb.VFS()
            with vfs.open(source_uri, "rb") as f:
                dimensions = int.from_bytes(f.read(4), "little")
                vector_size = 4 + dimensions * 4
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                size = int(file_size / vector_size)
                return size, dimensions, np.int32
        elif source_type == "BVEC":
            vfs = tiledb.VFS()
            with vfs.open(source_uri, "rb") as f:
                dimensions = int.from_bytes(f.read(4), "little")
                vector_size = 4 + dimensions
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                size = int(file_size / vector_size)
                return size, dimensions, np.uint8
        else:
            raise ValueError(
                f"Not supported source_type {source_type} - valid types are [TILEDB_ARRAY, TILEDB_SPARSE_ARRAY, U8BIN, F32BIN, FVEC, IVEC, BVEC]"
            )

    def create_array(
        group: tiledb.Group,
        size: int,
        dimensions: int,
        vector_type: np.dtype,
        array_name: str,
    ) -> str:
        input_vectors_array_uri = f"{group.uri}/{array_name}"
        if tiledb.array_exists(input_vectors_array_uri):
            raise ValueError(f"Array exists {input_vectors_array_uri}")
        tile_size = min(
            size,
            int(
                flat_index.TILE_SIZE_BYTES / np.dtype(vector_type).itemsize / dimensions
            ),
        )

        logger.debug("Creating input vectors array")
        input_vectors_array_rows_dim = tiledb.Dim(
            name="rows",
            domain=(0, dimensions - 1),
            tile=dimensions,
            dtype=np.dtype(np.int32),
        )
        input_vectors_array_cols_dim = tiledb.Dim(
            name="cols",
            domain=(0, size - 1),
            tile=tile_size,
            dtype=np.dtype(np.int32),
        )
        input_vectors_array_dom = tiledb.Domain(
            input_vectors_array_rows_dim, input_vectors_array_cols_dim
        )
        input_vectors_array_attr = tiledb.Attr(
            name="values", dtype=vector_type, filters=DEFAULT_ATTR_FILTERS
        )
        input_vectors_array_schema = tiledb.ArraySchema(
            domain=input_vectors_array_dom,
            sparse=False,
            attrs=[input_vectors_array_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        logger.debug(input_vectors_array_schema)
        tiledb.Array.create(input_vectors_array_uri, input_vectors_array_schema)
        add_to_group(group, input_vectors_array_uri, array_name)

        return input_vectors_array_uri

    def write_input_vectors(
        group: tiledb.Group,
        input_vectors: np.ndarray,
        size: int,
        dimensions: int,
        vector_type: np.dtype,
        array_name: str,
    ) -> str:
        input_vectors_array_uri = create_array(
            group=group,
            size=size,
            dimensions=dimensions,
            vector_type=vector_type,
            array_name=array_name,
        )

        input_vectors_array = tiledb.open(
            input_vectors_array_uri, "w", timestamp=index_timestamp
        )
        input_vectors_array[:, :] = np.transpose(input_vectors)
        input_vectors_array.close()

        return input_vectors_array_uri

    def write_external_ids(
        group: tiledb.Group,
        external_ids: np.array,
        size: int,
        partitions: int,
    ) -> str:
        external_ids_array_uri = f"{group.uri}/{EXTERNAL_IDS_ARRAY_NAME}"
        if tiledb.array_exists(external_ids_array_uri):
            raise ValueError(f"Array exists {external_ids_array_uri}")

        logger.debug("Creating external IDs array")
        ids_array_rows_dim = tiledb.Dim(
            name="rows",
            domain=(0, size - 1),
            tile=int(size / partitions),
            dtype=np.dtype(np.int32),
        )
        ids_array_dom = tiledb.Domain(ids_array_rows_dim)
        ids_attr = tiledb.Attr(
            name="values",
            dtype=np.dtype(np.uint64),
            filters=DEFAULT_ATTR_FILTERS,
        )
        ids_schema = tiledb.ArraySchema(
            domain=ids_array_dom,
            sparse=False,
            attrs=[ids_attr],
            capacity=int(size / partitions),
            cell_order="col-major",
            tile_order="col-major",
        )
        logger.debug(ids_schema)
        tiledb.Array.create(external_ids_array_uri, ids_schema)
        add_to_group(group, external_ids_array_uri, EXTERNAL_IDS_ARRAY_NAME)

        external_ids_array = tiledb.open(
            external_ids_array_uri, "w", timestamp=index_timestamp
        )
        external_ids_array[:] = external_ids
        external_ids_array.close()

        return external_ids_array_uri

    def create_temp_data_group(
        group: tiledb.Group,
    ) -> tiledb.Group:
        partial_write_array_dir_uri = f"{group.uri}/{PARTIAL_WRITE_ARRAY_DIR}"
        try:
            tiledb.group_create(partial_write_array_dir_uri)
            add_to_group(group, partial_write_array_dir_uri, PARTIAL_WRITE_ARRAY_DIR)
        except tiledb.TileDBError as err:
            message = str(err)
            if "already exists" not in message:
                raise err
        return tiledb.Group(partial_write_array_dir_uri, "w")

    def create_partial_write_array_group(
        temp_data_group: tiledb.Group,
        vector_type: np.dtype,
        dimensions: int,
        filters: Any,
        create_index_array: bool,
    ) -> str:
        tile_size = int(
            ivf_flat_index.TILE_SIZE_BYTES / np.dtype(vector_type).itemsize / dimensions
        )
        partial_write_array_index_uri = f"{temp_data_group.uri}/{INDEX_ARRAY_NAME}"
        partial_write_array_ids_uri = f"{temp_data_group.uri}/{IDS_ARRAY_NAME}"
        partial_write_array_parts_uri = f"{temp_data_group.uri}/{PARTS_ARRAY_NAME}"
        if create_index_array:
            try:
                tiledb.group_create(partial_write_array_index_uri)
            except tiledb.TileDBError as err:
                message = str(err)
                if "already exists" in message:
                    logger.debug(
                        f"Group '{partial_write_array_index_uri}' already exists"
                    )
                raise err
            add_to_group(
                temp_data_group,
                partial_write_array_index_uri,
                INDEX_ARRAY_NAME,
            )

        if not tiledb.array_exists(partial_write_array_ids_uri):
            logger.debug("Creating temp ids array")
            ids_array_rows_dim = tiledb.Dim(
                name="rows",
                domain=(0, MAX_INT32),
                tile=tile_size,
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
                capacity=tile_size,
                cell_order="col-major",
                tile_order="col-major",
            )
            logger.debug(ids_schema)
            tiledb.Array.create(partial_write_array_ids_uri, ids_schema)
            add_to_group(
                temp_data_group,
                partial_write_array_ids_uri,
                IDS_ARRAY_NAME,
            )

        if not tiledb.array_exists(partial_write_array_parts_uri):
            logger.debug("Creating temp parts array")
            parts_array_rows_dim = tiledb.Dim(
                name="rows",
                domain=(0, dimensions - 1),
                tile=dimensions,
                dtype=np.dtype(np.int32),
            )
            parts_array_cols_dim = tiledb.Dim(
                name="cols",
                domain=(0, MAX_INT32),
                tile=tile_size,
                dtype=np.dtype(np.int32),
            )
            parts_array_dom = tiledb.Domain(parts_array_rows_dim, parts_array_cols_dim)
            parts_attr = tiledb.Attr(name="values", dtype=vector_type, filters=filters)
            parts_schema = tiledb.ArraySchema(
                domain=parts_array_dom,
                sparse=False,
                attrs=[parts_attr],
                cell_order="col-major",
                tile_order="col-major",
            )
            logger.debug(parts_schema)
            logger.debug(partial_write_array_parts_uri)
            tiledb.Array.create(partial_write_array_parts_uri, parts_schema)
            add_to_group(
                temp_data_group,
                partial_write_array_parts_uri,
                PARTS_ARRAY_NAME,
            )
        return partial_write_array_index_uri

    def create_arrays(
        group: tiledb.Group,
        temp_data_group: tiledb.Group,
        arrays_created: bool,
        index_type: str,
        dimensions: int,
        input_vectors_work_items: int,
        vector_type: np.dtype,
        logger: logging.Logger,
        storage_version: str,
    ) -> None:
        if index_type == "FLAT":
            if not arrays_created:
                flat_index.create(
                    uri=group.uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    group_exists=True,
                    config=config,
                    storage_version=storage_version,
                )
        elif index_type == "IVF_FLAT":
            if not arrays_created:
                ivf_flat_index.create(
                    uri=group.uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    group_exists=True,
                    config=config,
                    storage_version=storage_version,
                )
            partial_write_array_index_uri = create_partial_write_array_group(
                temp_data_group=temp_data_group,
                vector_type=vector_type,
                dimensions=dimensions,
                filters=DEFAULT_ATTR_FILTERS,
                create_index_array=True,
            )
            partial_write_array_index_group = tiledb.Group(
                partial_write_array_index_uri, "w"
            )

            for part in range(input_vectors_work_items):
                part_index_uri = partial_write_array_index_uri + "/" + str(part)
                if not tiledb.array_exists(part_index_uri):
                    logger.debug(f"Creating part array {part_index_uri}")
                    index_array_rows_dim = tiledb.Dim(
                        name="rows",
                        domain=(0, partitions),
                        tile=partitions,
                        dtype=np.dtype(np.int32),
                    )
                    index_array_dom = tiledb.Domain(index_array_rows_dim)
                    index_attr = tiledb.Attr(
                        name="values",
                        dtype=np.dtype(np.uint64),
                        filters=DEFAULT_ATTR_FILTERS,
                    )
                    index_schema = tiledb.ArraySchema(
                        domain=index_array_dom,
                        sparse=False,
                        attrs=[index_attr],
                        capacity=partitions,
                        cell_order="col-major",
                        tile_order="col-major",
                    )
                    logger.debug(index_schema)
                    tiledb.Array.create(part_index_uri, index_schema)
                    add_to_group(
                        partial_write_array_index_group, part_index_uri, str(part)
                    )
            if updates_uri is not None:
                part_index_uri = partial_write_array_index_uri + "/additions"
                if not tiledb.array_exists(part_index_uri):
                    logger.debug(f"Creating part array {part_index_uri}")
                    index_array_rows_dim = tiledb.Dim(
                        name="rows",
                        domain=(0, partitions),
                        tile=partitions,
                        dtype=np.dtype(np.int32),
                    )
                    index_array_dom = tiledb.Domain(index_array_rows_dim)
                    index_attr = tiledb.Attr(
                        name="values",
                        dtype=np.dtype(np.uint64),
                        filters=DEFAULT_ATTR_FILTERS,
                    )
                    index_schema = tiledb.ArraySchema(
                        domain=index_array_dom,
                        sparse=False,
                        attrs=[index_attr],
                        capacity=partitions,
                        cell_order="col-major",
                        tile_order="col-major",
                    )
                    logger.debug(index_schema)
                    tiledb.Array.create(part_index_uri, index_schema)
                    add_to_group(
                        partial_write_array_index_group, part_index_uri, "additions"
                    )
            partial_write_array_index_group.close()

        # Note that we don't create type-erased indexes (i.e. Vamana) here. Instead we create them
        # at very start of ingest() in C++.
        elif not is_type_erased_index(index_type):
            raise ValueError(f"Not supported index_type {index_type}")

    def read_external_ids(
        external_ids_uri: str,
        external_ids_type: str,
        start_pos: int,
        end_pos: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ) -> np.array:
        logger = setup(config, verbose)
        logger.debug(
            "Reading external_ids start_pos: %i, end_pos: %i", start_pos, end_pos
        )
        if external_ids_uri == "":
            return np.arange(start_pos, end_pos).astype(np.uint64)
        if external_ids_type == "TILEDB_ARRAY":
            with tiledb.open(
                external_ids_uri, mode="r", timestamp=index_timestamp
            ) as external_ids_array:
                return external_ids_array[start_pos:end_pos]["values"]
        elif source_type == "TILEDB_PARTITIONED_ARRAY":
            with tiledb.open(source_uri, "r") as source_array:
                q = source_array.query(attrs=("vectors_shape",), coords=True)
                nonempty_object_array_domain = source_array.nonempty_domain()
                partitions = q[
                    nonempty_object_array_domain[0][0] : nonempty_object_array_domain[
                        0
                    ][1]
                    + 1
                ]
                partition_idx_start = 0
                partition_idx_end = 0
                i = 0
                external_ids = None
                for partition_shape in partitions["vectors_shape"]:
                    partition_id = partitions["partition_id"][i]
                    partition_idx_end += partition_shape[0]
                    intersection_start = max(start_pos, partition_idx_start)
                    intersection_end = min(end_pos, partition_idx_end)
                    if intersection_start < intersection_end:
                        crop_start = intersection_start - partition_idx_start
                        crop_end = intersection_end - partition_idx_start
                        qv = source_array.query(attrs=("external_ids",), coords=True)
                        partition_external_ids = qv[partition_id : partition_id + 1][
                            "external_ids"
                        ][0][crop_start:crop_end]
                        if external_ids is None:
                            external_ids = partition_external_ids
                        else:
                            external_ids = np.concatenate(
                                (external_ids, partition_external_ids)
                            )
                    partition_idx_start = partition_idx_end
                    i += 1
            return external_ids
        elif external_ids_type == "U64BIN":
            vfs = tiledb.VFS()
            read_size = end_pos - start_pos
            read_offset = start_pos + 8
            with vfs.open(external_ids_uri, "rb") as f:
                f.seek(read_offset)
                return np.reshape(
                    np.frombuffer(
                        f.read(read_size),
                        count=read_size,
                        dtype=np.uint64,
                    ).astype(np.uint64),
                    (read_size),
                )

    def read_additions(
        updates_uri: str,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ) -> (np.ndarray, np.array):
        if updates_uri is None:
            return None, None
        logger = setup(config, verbose)
        logger.debug("Reading additions vectors")
        with tiledb.open(
            updates_uri,
            mode="r",
            timestamp=(previous_ingestion_timestamp, index_timestamp),
        ) as updates_array:
            q = updates_array.query(attrs=("vector",), coords=True)
            data = q[:]
            additions_filter = [len(item) > 0 for item in data["vector"]]
            filtered_vectors = data["vector"][additions_filter]
            if len(filtered_vectors) == 0:
                return None, None
            else:
                return (
                    np.vstack(data["vector"][additions_filter]),
                    data["external_id"][additions_filter],
                )

    def read_updated_ids(
        updates_uri: str,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ) -> np.array:
        if updates_uri is None:
            return np.array([], np.uint64)
        logger = setup(config, verbose)
        logger.debug("Reading updated vector ids")
        with tiledb.open(
            updates_uri,
            mode="r",
            timestamp=(previous_ingestion_timestamp, index_timestamp),
        ) as updates_array:
            q = updates_array.query(attrs=("vector",), coords=True)
            data = q[:]
            return data["external_id"]

    def read_input_vectors(
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        dimensions: int,
        start_pos: int,
        end_pos: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ) -> np.ndarray:
        logger = setup(config, verbose)
        logger.debug(
            "Reading input vectors start_pos: %i, end_pos: %i", start_pos, end_pos
        )
        if source_type == "TILEDB_ARRAY":
            with tiledb.open(
                source_uri, mode="r", timestamp=index_timestamp
            ) as src_array:
                src_array_schema = src_array.schema
                return np.transpose(
                    src_array[0:dimensions, start_pos:end_pos][
                        src_array_schema.attr(0).name
                    ]
                ).copy(order="C")
        if source_type == "TILEDB_SPARSE_ARRAY":
            from scipy.sparse import coo_matrix

            with tiledb.open(
                source_uri, mode="r", timestamp=index_timestamp
            ) as src_array:
                src_array_schema = src_array.schema
                data = src_array[start_pos:end_pos, 0:dimensions]
                return coo_matrix(
                    (
                        data[src_array_schema.attr(0).name],
                        (
                            data[src_array_schema.domain.dim(0).name] - start_pos,
                            data[src_array_schema.domain.dim(1).name],
                        ),
                    )
                ).toarray()
        elif source_type == "TILEDB_PARTITIONED_ARRAY":
            with tiledb.open(
                source_uri, "r", timestamp=index_timestamp, config=config
            ) as source_array:
                q = source_array.query(attrs=("vectors_shape",), coords=True)
                nonempty_object_array_domain = source_array.nonempty_domain()
                partitions = q[
                    nonempty_object_array_domain[0][0] : nonempty_object_array_domain[
                        0
                    ][1]
                    + 1
                ]
                partition_idx_start = 0
                partition_idx_end = 0
                i = 0
                vectors = None
                for partition_shape in partitions["vectors_shape"]:
                    partition_id = partitions["partition_id"][i]
                    partition_idx_end += partition_shape[0]
                    intersection_start = max(start_pos, partition_idx_start)
                    intersection_end = min(end_pos, partition_idx_end)
                    if intersection_start < intersection_end:
                        crop_start = intersection_start - partition_idx_start
                        crop_end = intersection_end - partition_idx_start
                        qv = source_array.query(attrs=("vectors",), coords=True)
                        partition_vectors = np.reshape(
                            qv[partition_id : partition_id + 1]["vectors"][0],
                            partition_shape,
                        )[crop_start:crop_end]
                        if vectors is None:
                            vectors = partition_vectors
                        else:
                            vectors = np.concatenate((vectors, partition_vectors))
                    partition_idx_start = partition_idx_end
                    i += 1
            return vectors
        elif source_type == "U8BIN":
            vfs = tiledb.VFS()
            read_size = end_pos - start_pos
            read_offset = start_pos * dimensions + 8
            with vfs.open(source_uri, "rb") as f:
                f.seek(read_offset)
                return np.reshape(
                    np.frombuffer(
                        f.read(read_size * dimensions),
                        count=read_size * dimensions,
                        dtype=vector_type,
                    ).astype(vector_type),
                    (read_size, dimensions),
                )
        elif source_type == "F32BIN":
            vfs = tiledb.VFS()
            read_size = end_pos - start_pos
            read_offset = start_pos * dimensions * 4 + 8
            with vfs.open(source_uri, "rb") as f:
                f.seek(read_offset)
                return np.reshape(
                    np.frombuffer(
                        f.read(read_size * dimensions * 4),
                        count=read_size * dimensions,
                        dtype=vector_type,
                    ).astype(vector_type),
                    (read_size, dimensions),
                )
        elif source_type == "FVEC" or source_type == "IVEC":
            vfs = tiledb.VFS()
            vector_values = 1 + dimensions
            vector_size = vector_values * 4
            read_size = end_pos - start_pos
            read_offset = start_pos * vector_size
            with vfs.open(source_uri, "rb") as f:
                f.seek(read_offset)
                return np.delete(
                    np.reshape(
                        np.frombuffer(
                            f.read(read_size * vector_size),
                            count=read_size * vector_values,
                            dtype=vector_type,
                        ).astype(vector_type),
                        (read_size, dimensions + 1),
                    ),
                    0,
                    axis=1,
                )
        elif source_type == "BVEC":
            vfs = tiledb.VFS()
            vector_values = 1 + dimensions
            vector_size = vector_values * 1
            read_size = end_pos - start_pos
            read_offset = start_pos * vector_size
            with vfs.open(source_uri, "rb") as f:
                f.seek(read_offset)
                return np.delete(
                    np.reshape(
                        np.frombuffer(
                            f.read(read_size * vector_size),
                            count=read_size * vector_values,
                            dtype=vector_type,
                        ).astype(vector_type),
                        (read_size, dimensions + 1),
                    ),
                    0,
                    axis=1,
                )

    # --------------------------------------------------------------------
    # UDFs
    # --------------------------------------------------------------------

    def copy_centroids(
        index_group_uri: str,
        copy_centroids_uri: str,
        partitions: int,
        dimensions: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        logger = setup(config, verbose)
        group = tiledb.Group(index_group_uri)
        centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
        logger.debug(
            "Copying centroids from: %s, to: %s", copy_centroids_uri, centroids_uri
        )
        src = tiledb.open(copy_centroids_uri, mode="r")
        dest = tiledb.open(centroids_uri, mode="w", timestamp=index_timestamp)
        src_centroids = src[0:dimensions, 0:partitions]
        dest[0:dimensions, 0:partitions] = src_centroids
        logger.debug(src_centroids)

    # --------------------------------------------------------------------
    # centralised kmeans UDFs
    # --------------------------------------------------------------------
    def random_sample_from_input_vectors(
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        dimensions: int,
        source_start_pos: int,
        source_end_pos: int,
        batch: int,
        random_sample_size: int,
        output_source_uri: str,
        output_start_pos: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
    ):
        """
        Reads a random sample of vectors from the source data and appends them to the output array.

        Parameters
        ----------
        source_uri: str
            Data source URI.
        source_type: str
            Type of the source data.
        vector_type: np.dtype
            Type of the vectors.
        dimensions: int
            Number of dimensions in a vector.
        vector_start_pos: int
            Start position of source_uri to read from.
        vector_end_pos: int
            End position of source_uri to read to.
        batch: int
            Read the source data in batches of this size.
        random_sample_size: int
            Number of vectors to randomly sample from the source data.
        output_source_uri: str
            URI of the output array.
        output_start_pos: int
            Start position of the output array to write to.
        """
        if random_sample_size == 0:
            return

        with tiledb.scope_ctx(ctx_or_config=config):
            source_size = source_end_pos - source_start_pos
            num_sampled = 0
            for start in range(source_start_pos, source_end_pos, batch):
                # What vectors to read from the source_uri.
                end = start + batch
                if end > source_end_pos:
                    end = source_end_pos

                # How many vectors sample from the vectors read.
                percent_of_data_to_read = (end - start) / source_size
                num_to_sample = math.ceil(random_sample_size * percent_of_data_to_read)
                if num_sampled + num_to_sample > random_sample_size:
                    num_to_sample = random_sample_size - num_sampled
                if num_to_sample == 0:
                    continue
                num_sampled += num_to_sample

                # Read from the source data.
                vectors = read_input_vectors(
                    source_uri=source_uri,
                    source_type=source_type,
                    vector_type=vector_type,
                    dimensions=dimensions,
                    start_pos=start,
                    end_pos=end,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                )

                # Randomly sample from the data we read.
                row_indices = np.random.choice(
                    vectors.shape[0], size=num_to_sample, replace=False
                )
                sampled_vectors = vectors[row_indices]

                # Append to output array.
                with tiledb.open(
                    output_source_uri, mode="w", timestamp=index_timestamp
                ) as A:
                    A[
                        0:dimensions,
                        output_start_pos : output_start_pos + num_to_sample,
                    ] = np.transpose(sampled_vectors)

        if num_sampled != random_sample_size:
            raise ValueError(
                f"The random sampling within a batch ran into an issue: num_sampled ({num_sampled}) != random_sample_size ({random_sample_size})"
            )

    def centralised_kmeans(
        index_group_uri: str,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        partitions: int,
        dimensions: int,
        training_sample_size: int,
        training_source_uri: Optional[str],
        training_source_type: Optional[str],
        init: str = "random",
        max_iter: int = 10,
        n_init: int = 1,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        use_sklearn: bool = True,
    ):
        from sklearn.cluster import KMeans

        from tiledb.vector_search.module import array_to_matrix

        with tiledb.scope_ctx(ctx_or_config=config):
            logger = setup(config, verbose)
            group = tiledb.Group(index_group_uri)
            centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
            if training_sample_size >= partitions:
                if training_source_uri:
                    if training_source_type is None:
                        training_source_type = autodetect_source_type(
                            source_uri=training_source_uri
                        )
                    (
                        training_in_size,
                        training_dimensions,
                        training_vector_type,
                    ) = read_source_metadata(
                        source_uri=training_source_uri, source_type=training_source_type
                    )
                    if dimensions != training_dimensions:
                        raise ValueError(
                            f"When training centroids, the index data dimensions ({dimensions}) != the training data dimensions ({training_dimensions})"
                        )
                    sample_vectors = read_input_vectors(
                        source_uri=training_source_uri,
                        source_type=training_source_type,
                        vector_type=training_vector_type,
                        dimensions=dimensions,
                        start_pos=0,
                        end_pos=training_in_size,
                        config=config,
                        verbose=verbose,
                        trace_id=trace_id,
                    ).astype(np.float32)
                else:
                    sample_vectors = read_input_vectors(
                        source_uri=source_uri,
                        source_type=source_type,
                        vector_type=vector_type,
                        dimensions=dimensions,
                        start_pos=0,
                        end_pos=training_sample_size,
                        config=config,
                        verbose=verbose,
                        trace_id=trace_id,
                    ).astype(np.float32)

                logger.debug("Start kmeans training")
                if use_sklearn:
                    km = KMeans(
                        n_clusters=partitions,
                        init=init,
                        max_iter=max_iter,
                        verbose=3 if verbose else 0,
                        n_init=n_init,
                        random_state=0,
                    )
                    km.fit_predict(sample_vectors)
                    centroids = np.transpose(np.array(km.cluster_centers_))
                else:
                    from tiledb.vector_search.module import kmeans_fit

                    centroids = kmeans_fit(
                        partitions,
                        init,
                        max_iter,
                        verbose,
                        n_init,
                        array_to_matrix(np.transpose(sample_vectors)),
                    )
                    centroids = np.array(centroids)  # TODO: why is this here?
            else:
                # TODO(paris): Should we instead take the first training_sample_size vectors and then fill in random for the rest? Or raise an error like this:
                # raise ValueError(f"We have a training_sample_size of {training_sample_size} but {partitions} partitions - training_sample_size must be >= partitions")
                centroids = np.random.rand(dimensions, partitions)

            logger.debug("Writing centroids to array %s", centroids_uri)
            with tiledb.open(centroids_uri, mode="w", timestamp=index_timestamp) as A:
                A[0:dimensions, 0:partitions] = centroids

    # --------------------------------------------------------------------
    # distributed kmeans UDFs
    # --------------------------------------------------------------------
    def init_centroids(
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        partitions: int,
        dimensions: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ) -> np.ndarray:
        logger = setup(config, verbose)
        logger.debug(
            "Initialising centroids by reading the first vectors in the source data."
        )
        with tiledb.scope_ctx(ctx_or_config=config):
            return read_input_vectors(
                source_uri=source_uri,
                source_type=source_type,
                vector_type=vector_type,
                dimensions=dimensions,
                start_pos=0,
                end_pos=partitions,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
            )

    def assign_points_and_partial_new_centroids(
        centroids: np.ndarray,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        partitions: int,
        dimensions: int,
        vector_start_pos: int,
        vector_end_pos: int,
        threads: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        use_sklearn: bool = True,
    ):
        from sklearn.cluster import KMeans

        import tiledb.cloud

        def generate_new_centroid_per_thread(
            thread_id, start, end, new_centroid_sums_queue, new_centroid_counts_queue
        ):
            new_centroid_sums = []
            for i in range(len(cents_t)):
                new_centroid_sums.append(cents_t[i])
            new_centroid_count = np.ones(len(cents_t))
            for vector_id in range(start, end):
                if vector_id % 100000 == 0:
                    logger.debug("Vectors computed: %d", vector_id)
                c_id = assignments_t[vector_id]
                if new_centroid_count[c_id] == 1:
                    new_centroid_sums[c_id] = vectors_t[vector_id]
                else:
                    for d in range(dimensions):
                        new_centroid_sums[c_id][d] += vectors_t[vector_id][d]
                new_centroid_count[c_id] += 1
            new_centroid_sums_queue.put(new_centroid_sums)
            new_centroid_counts_queue.put(new_centroid_count)
            logger.debug("Finished thread: %d", thread_id)

        def update_centroids():
            import multiprocessing as mp

            logger.debug("Updating centroids based on assignments.")
            logger.debug("Using %d threads.", threads)
            global cents_t, vectors_t, assignments_t, new_centroid_thread_sums, new_centroid_thread_counts
            cents_t = centroids
            vectors_t = vectors
            assignments_t = assignments
            new_centroid_thread_sums = []
            new_centroid_thread_counts = []
            workers = []
            thread_id = 0
            batch_size = math.ceil(len(vectors) / threads)

            for i in range(0, len(vectors), batch_size):
                new_centroid_sums_queue = mp.Queue()
                new_centroid_thread_sums.append(new_centroid_sums_queue)
                new_centroid_counts_queue = mp.Queue()
                new_centroid_thread_counts.append(new_centroid_counts_queue)

                start = i
                end = i + batch_size
                if end > len(vectors):
                    end = len(vectors)
                worker = mp.Process(
                    target=generate_new_centroid_per_thread,
                    args=(
                        thread_id,
                        start,
                        end,
                        new_centroid_sums_queue,
                        new_centroid_counts_queue,
                    ),
                )
                worker.start()
                workers.append(worker)
                thread_id += 1

            new_centroid_thread_sums_array = []
            new_centroid_thread_counts_array = []
            for i in range(threads):
                new_centroid_thread_sums_array.append(new_centroid_thread_sums[i].get())
                new_centroid_thread_counts_array.append(
                    new_centroid_thread_counts[i].get()
                )
                workers[i].join()

            logger.debug("Finished all threads, aggregating partial results.")
            new_centroids = []
            for c_id in range(partitions):
                cent = []
                for d in range(dimensions):
                    sum = 0
                    count = 0
                    for thread_id in range(threads):
                        sum += new_centroid_thread_sums_array[thread_id][c_id][d]
                        count += new_centroid_thread_counts_array[thread_id][c_id]
                    cent.append(sum / count)
                new_centroids.append(cent)
            return new_centroids

        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            logger.debug("Reading input vectors.")
            vectors = read_input_vectors(
                source_uri=source_uri,
                source_type=source_type,
                vector_type=vector_type,
                dimensions=dimensions,
                start_pos=vector_start_pos,
                end_pos=vector_end_pos,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
            )
            logger.debug("Input centroids: %s", centroids[0:5])
            logger.debug("Assigning vectors to centroids")
            if use_sklearn:
                km = KMeans()
                km._n_threads = threads
                km.cluster_centers_ = centroids
                assignments = km.predict(vectors)
            else:
                assignments = kmeans_predict(centroids, vectors)
            logger.debug("Assignments: %s", assignments[0:100])
            partial_new_centroids = update_centroids()
            logger.debug("New centroids: %s", partial_new_centroids[0:5])
            return partial_new_centroids

    def compute_new_centroids(*argv):
        import numpy as np

        return np.mean(argv, axis=0).astype(np.float32)

    def ingest_flat(
        index_group_uri: str,
        source_uri: str,
        source_type: str,
        updates_uri: str,
        vector_type: np.dtype,
        external_ids_uri: str,
        external_ids_type: str,
        dimensions: int,
        size: int,
        batch: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        import numpy as np

        import tiledb.cloud

        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            updated_ids = read_updated_ids(
                updates_uri=updates_uri,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
            )
            group = tiledb.Group(index_group_uri)
            parts_array_uri = group[PARTS_ARRAY_NAME].uri
            ids_array_uri = group[IDS_ARRAY_NAME].uri
            parts_array = tiledb.open(
                parts_array_uri, mode="w", timestamp=index_timestamp
            )
            ids_array = tiledb.open(ids_array_uri, mode="w", timestamp=index_timestamp)
            # Ingest base data
            write_offset = 0
            for part in range(0, size, batch):
                part_end = part + batch
                if part_end > size:
                    part_end = size
                in_vectors = read_input_vectors(
                    source_uri=source_uri,
                    source_type=source_type,
                    vector_type=vector_type,
                    dimensions=dimensions,
                    start_pos=part,
                    end_pos=part_end,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                )
                external_ids = read_external_ids(
                    external_ids_uri=external_ids_uri,
                    external_ids_type=external_ids_type,
                    start_pos=part,
                    end_pos=part_end,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                )
                updates_filter = np.in1d(
                    external_ids, updated_ids, assume_unique=True, invert=True
                )
                in_vectors = in_vectors[updates_filter]
                external_ids = external_ids[updates_filter]
                vector_len = len(in_vectors)
                if vector_len > 0:
                    end_offset = write_offset + vector_len
                    logger.debug("Vector read: %d", vector_len)
                    logger.debug("Writing input data to array %s", parts_array_uri)
                    parts_array[0:dimensions, write_offset:end_offset] = np.transpose(
                        in_vectors
                    )
                    logger.debug("Writing input data to array %s", ids_array_uri)
                    ids_array[write_offset:end_offset] = external_ids
                    write_offset = end_offset

            # Ingest additions
            additions_vectors, additions_external_ids = read_additions(
                updates_uri=updates_uri,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
            )
            end = write_offset
            if additions_vectors is not None:
                end += len(additions_external_ids)
                logger.debug("Writing additions data to array %s", parts_array_uri)
                parts_array[0:dimensions, write_offset:end] = np.transpose(
                    additions_vectors
                )
                logger.debug("Writing additions  data to array %s", ids_array_uri)
                ids_array[write_offset:end] = additions_external_ids
            group = tiledb.Group(index_group_uri, "w")
            group.meta["temp_size"] = end
            group.close()
            parts_array.close()
            ids_array.close()

    def ingest_type_erased(
        index_type: str,
        index_group_uri: str,
        source_uri: str,
        source_type: str,
        updates_uri: str,
        vector_type: np.dtype,
        external_ids_uri: str,
        external_ids_type: str,
        dimensions: int,
        size: int,
        batch: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        import numpy as np

        import tiledb.cloud
        from tiledb.vector_search.storage_formats import storage_formats

        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            updated_ids = read_updated_ids(
                updates_uri=updates_uri,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
            )

            temp_data_group_uri = f"{index_group_uri}/{PARTIAL_WRITE_ARRAY_DIR}"
            temp_data_group = tiledb.Group(temp_data_group_uri, "w")
            create_partial_write_array_group(
                temp_data_group=temp_data_group,
                vector_type=vector_type,
                dimensions=dimensions,
                filters=storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"],
                create_index_array=False,
            )
            temp_data_group.close()
            temp_data_group = tiledb.Group(temp_data_group_uri)
            ids_array_uri = temp_data_group[IDS_ARRAY_NAME].uri
            parts_array_uri = temp_data_group[PARTS_ARRAY_NAME].uri
            temp_data_group.close()

            parts_array = tiledb.open(
                parts_array_uri, mode="w", timestamp=index_timestamp
            )
            ids_array = tiledb.open(ids_array_uri, mode="w", timestamp=index_timestamp)
            # Ingest base data
            write_offset = 0
            for part in range(0, size, batch):
                part_end = part + batch
                if part_end > size:
                    part_end = size
                # First we get each vector and it's external id from the input data.
                in_vectors = read_input_vectors(
                    source_uri=source_uri,
                    source_type=source_type,
                    vector_type=vector_type,
                    dimensions=dimensions,
                    start_pos=part,
                    end_pos=part_end,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                )
                external_ids = read_external_ids(
                    external_ids_uri=external_ids_uri,
                    external_ids_type=external_ids_type,
                    start_pos=part,
                    end_pos=part_end,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                )

                # Then check if the external id is in the updated ids.
                updates_filter = np.in1d(
                    external_ids, updated_ids, assume_unique=True, invert=True
                )
                # We only keep the vectors and external ids that are not in the updated ids.
                in_vectors = in_vectors[updates_filter]
                external_ids = external_ids[updates_filter]
                vector_len = len(in_vectors)
                if vector_len > 0:
                    end_offset = write_offset + vector_len
                    logger.debug("Vector read: %d", vector_len)
                    logger.debug("Writing input data to array %s", parts_array_uri)
                    # Write the not-updated vectors to the parts array.
                    parts_array[0:dimensions, write_offset:end_offset] = np.transpose(
                        in_vectors
                    )
                    logger.debug("Writing input data to array %s", ids_array_uri)
                    # Write the not-updated external ids to the ids array.
                    ids_array[write_offset:end_offset] = external_ids
                    write_offset = end_offset

            # Ingest additions
            additions_vectors, additions_external_ids = read_additions(
                updates_uri=updates_uri,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
            )
            end = write_offset
            if additions_vectors is not None:
                end += len(additions_external_ids)
                logger.debug("Writing additions data to array %s", parts_array_uri)
                parts_array[0:dimensions, write_offset:end] = np.transpose(
                    additions_vectors
                )
                logger.debug("Writing additions  data to array %s", ids_array_uri)
                ids_array[write_offset:end] = additions_external_ids

            group = tiledb.Group(index_group_uri, "w")
            group.meta["temp_size"] = end
            group.close()

            parts_array.close()
            ids_array.close()

        # Now that we've ingested the vectors and their IDs, train the index with the data.
        from tiledb.vector_search import _tiledbvspy as vspy

        ctx = vspy.Ctx(config)
        if index_type == "VAMANA":
            index = vspy.IndexVamana(ctx, index_group_uri)
        elif index_type == "IVF_PQ":
            index = vspy.IndexIVFPQ(ctx, index_group_uri)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        data = vspy.FeatureVectorArray(
            ctx, parts_array_uri, ids_array_uri, 0, to_temporal_policy(index_timestamp)
        )
        index.train(data)
        index.add(data)
        index.write_index(ctx, index_group_uri, to_temporal_policy(index_timestamp))

    def write_centroids(
        centroids: np.ndarray,
        index_group_uri: str,
        partitions: int,
        dimensions: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        with tiledb.scope_ctx(ctx_or_config=config):
            logger = setup(config, verbose)
            group = tiledb.Group(index_group_uri)
            centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
            logger.debug("Writing centroids to array %s", centroids_uri)
            with tiledb.open(centroids_uri, mode="w", timestamp=index_timestamp) as A:
                A[0:dimensions, 0:partitions] = np.transpose(np.array(centroids))

    # --------------------------------------------------------------------
    # vector ingestion UDFs
    # --------------------------------------------------------------------
    def ingest_vectors_udf(
        index_group_uri: str,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        external_ids_uri: str,
        external_ids_type: str,
        partitions: int,
        dimensions: int,
        start: int,
        end: int,
        batch: int,
        threads: int,
        updates_uri: str = None,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        import tiledb.cloud
        from tiledb.vector_search.module import StdVector_u64
        from tiledb.vector_search.module import array_to_matrix
        from tiledb.vector_search.module import ivf_index
        from tiledb.vector_search.module import ivf_index_tdb

        logger = setup(config, verbose)
        group = tiledb.Group(index_group_uri)
        centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
        partial_write_array_dir_uri = group[PARTIAL_WRITE_ARRAY_DIR].uri
        partial_write_array_group = tiledb.Group(partial_write_array_dir_uri)
        partial_write_array_ids_uri = partial_write_array_group[IDS_ARRAY_NAME].uri
        partial_write_array_parts_uri = partial_write_array_group[PARTS_ARRAY_NAME].uri
        partial_write_array_index_dir_uri = partial_write_array_group[
            INDEX_ARRAY_NAME
        ].uri
        partial_write_array_index_group = tiledb.Group(
            partial_write_array_index_dir_uri
        )

        for part in range(start, end, batch):
            part_end = part + batch
            if part_end > end:
                part_end = end

            str(part) + "-" + str(part_end)

            partial_write_array_index_uri = partial_write_array_index_group[
                str(int(part / batch))
            ].uri
            logger.debug("Input vectors start_pos: %d, end_pos: %d", part, part_end)
            updated_ids = read_updated_ids(
                updates_uri=updates_uri,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
            )
            if source_type == "TILEDB_ARRAY":
                logger.debug("Start indexing")
                if index_timestamp is None:
                    ivf_index_tdb(
                        dtype=vector_type,
                        db_uri=source_uri,
                        external_ids_uri=external_ids_uri,
                        deleted_ids=StdVector_u64(updated_ids),
                        centroids_uri=centroids_uri,
                        parts_uri=partial_write_array_parts_uri,
                        index_array_uri=partial_write_array_index_uri,
                        id_uri=partial_write_array_ids_uri,
                        start=part,
                        end=part_end,
                        nthreads=threads,
                        config=config,
                    )
                else:
                    ivf_index_tdb(
                        dtype=vector_type,
                        db_uri=source_uri,
                        external_ids_uri=external_ids_uri,
                        deleted_ids=StdVector_u64(updated_ids),
                        centroids_uri=centroids_uri,
                        parts_uri=partial_write_array_parts_uri,
                        index_array_uri=partial_write_array_index_uri,
                        id_uri=partial_write_array_ids_uri,
                        start=part,
                        end=part_end,
                        nthreads=threads,
                        timestamp=index_timestamp,
                        config=config,
                    )
            else:
                in_vectors = read_input_vectors(
                    source_uri=source_uri,
                    source_type=source_type,
                    vector_type=vector_type,
                    dimensions=dimensions,
                    start_pos=part,
                    end_pos=part_end,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                )
                external_ids = read_external_ids(
                    external_ids_uri=external_ids_uri,
                    external_ids_type=external_ids_type,
                    start_pos=part,
                    end_pos=part_end,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                )
                logger.debug("Start indexing")
                if index_timestamp is None:
                    ivf_index(
                        dtype=vector_type,
                        db=array_to_matrix(
                            np.transpose(in_vectors).astype(vector_type)
                        ),
                        external_ids=StdVector_u64(external_ids),
                        deleted_ids=StdVector_u64(updated_ids),
                        centroids_uri=centroids_uri,
                        parts_uri=partial_write_array_parts_uri,
                        index_array_uri=partial_write_array_index_uri,
                        id_uri=partial_write_array_ids_uri,
                        start=part,
                        end=part_end,
                        nthreads=threads,
                        config=config,
                    )
                else:
                    ivf_index(
                        dtype=vector_type,
                        db=array_to_matrix(
                            np.transpose(in_vectors).astype(vector_type)
                        ),
                        external_ids=StdVector_u64(external_ids),
                        deleted_ids=StdVector_u64(updated_ids),
                        centroids_uri=centroids_uri,
                        parts_uri=partial_write_array_parts_uri,
                        index_array_uri=partial_write_array_index_uri,
                        id_uri=partial_write_array_ids_uri,
                        start=part,
                        end=part_end,
                        nthreads=threads,
                        timestamp=index_timestamp,
                        config=config,
                    )

    def ingest_additions_udf(
        index_group_uri: str,
        updates_uri: str,
        vector_type: np.dtype,
        write_offset: int,
        threads: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        import tiledb.cloud
        from tiledb.vector_search.module import StdVector_u64
        from tiledb.vector_search.module import array_to_matrix
        from tiledb.vector_search.module import ivf_index

        logger = setup(config, verbose)
        group = tiledb.Group(index_group_uri)
        centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
        partial_write_array_dir_uri = group[PARTIAL_WRITE_ARRAY_DIR].uri
        partial_write_array_group = tiledb.Group(partial_write_array_dir_uri)
        partial_write_array_ids_uri = partial_write_array_group[IDS_ARRAY_NAME].uri
        partial_write_array_parts_uri = partial_write_array_group[PARTS_ARRAY_NAME].uri
        partial_write_array_index_dir_uri = partial_write_array_group[
            INDEX_ARRAY_NAME
        ].uri
        partial_write_array_index_group = tiledb.Group(
            partial_write_array_index_dir_uri
        )
        partial_write_array_index_uri = partial_write_array_index_group["additions"].uri
        additions_vectors, additions_external_ids = read_additions(
            updates_uri=updates_uri,
            config=config,
            verbose=verbose,
            trace_id=trace_id,
        )
        if additions_vectors is None:
            return

        logger.debug(f"Ingesting additions {partial_write_array_index_uri}")
        if index_timestamp is None:
            ivf_index(
                dtype=vector_type,
                db=array_to_matrix(np.transpose(additions_vectors).astype(vector_type)),
                external_ids=StdVector_u64(additions_external_ids),
                deleted_ids=StdVector_u64(np.array([], np.uint64)),
                centroids_uri=centroids_uri,
                parts_uri=partial_write_array_parts_uri,
                index_array_uri=partial_write_array_index_uri,
                id_uri=partial_write_array_ids_uri,
                start=write_offset,
                end=0,
                nthreads=threads,
                config=config,
            )
        else:
            ivf_index(
                dtype=vector_type,
                db=array_to_matrix(np.transpose(additions_vectors).astype(vector_type)),
                external_ids=StdVector_u64(additions_external_ids),
                deleted_ids=StdVector_u64(np.array([], np.uint64)),
                centroids_uri=centroids_uri,
                parts_uri=partial_write_array_parts_uri,
                index_array_uri=partial_write_array_index_uri,
                id_uri=partial_write_array_ids_uri,
                start=write_offset,
                end=0,
                nthreads=threads,
                timestamp=index_timestamp,
                config=config,
            )

    def compute_partition_indexes_udf(
        index_group_uri: str,
        partitions: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(index_group_uri)
            index_array_uri = group[INDEX_ARRAY_NAME].uri
            partial_write_array_dir_uri = group[PARTIAL_WRITE_ARRAY_DIR].uri
            partial_write_array_group = tiledb.Group(partial_write_array_dir_uri)
            partial_write_array_index_dir_uri = partial_write_array_group[
                INDEX_ARRAY_NAME
            ].uri
            partial_write_array_index_group = tiledb.Group(
                partial_write_array_index_dir_uri
            )
            partition_sizes = np.zeros(partitions)
            indexes = np.zeros(partitions + 1).astype(np.uint64)
            for part in partial_write_array_index_group:
                partial_index_array_uri = part.uri
                if tiledb.array_exists(partial_index_array_uri):
                    partial_index_array = tiledb.open(
                        partial_index_array_uri, mode="r", timestamp=index_timestamp
                    )
                    partial_indexes = partial_index_array[:]["values"]
                    i = 0
                    prev_index = partial_indexes[0]
                    for partial_index in partial_indexes[1:]:
                        partition_sizes[i] += int(partial_index) - int(prev_index)
                        prev_index = partial_index
                        i += 1
            logger.debug("Partition sizes: %s", partition_sizes)
            i = 0
            _sum = 0
            for partition_size in partition_sizes:
                indexes[i] = _sum
                _sum += partition_size
                i += 1
            indexes[i] = _sum
            group = tiledb.Group(index_group_uri, "w")
            group.meta["temp_size"] = _sum
            group.close()
            logger.debug(f"Partition indexes: {indexes}")
            index_array = tiledb.open(
                index_array_uri, mode="w", timestamp=index_timestamp
            )
            index_array[0 : partitions + 1] = indexes

    def consolidate_partition_udf(
        index_group_uri: str,
        partition_id_start: int,
        partition_id_end: int,
        batch: int,
        dimensions: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            logger.debug(
                "Consolidating partitions %d-%d", partition_id_start, partition_id_end
            )
            group = tiledb.Group(index_group_uri)
            partial_write_array_dir_uri = group[PARTIAL_WRITE_ARRAY_DIR].uri
            partial_write_array_group = tiledb.Group(partial_write_array_dir_uri)
            partial_write_array_ids_uri = partial_write_array_group[IDS_ARRAY_NAME].uri
            partial_write_array_parts_uri = partial_write_array_group[
                PARTS_ARRAY_NAME
            ].uri
            partial_write_array_index_dir_uri = partial_write_array_group[
                INDEX_ARRAY_NAME
            ].uri
            partial_write_array_index_group = tiledb.Group(
                partial_write_array_index_dir_uri
            )
            index_array_uri = group[INDEX_ARRAY_NAME].uri
            ids_array_uri = group[IDS_ARRAY_NAME].uri
            parts_array_uri = group[PARTS_ARRAY_NAME].uri
            partition_slices = []
            for i in range(partitions):
                partition_slices.append([])
            for part in partial_write_array_index_group:
                partial_index_array_uri = part.uri
                if tiledb.array_exists(partial_index_array_uri):
                    partial_index_array = tiledb.open(
                        partial_index_array_uri, mode="r", timestamp=index_timestamp
                    )
                    partial_indexes = partial_index_array[:]["values"]
                    prev_index = partial_indexes[0]
                    i = 0
                    for partial_index in partial_indexes[1:]:
                        s = slice(int(prev_index), int(partial_index - 1))
                        if (
                            s.start <= s.stop
                            and s.start != np.iinfo(np.dtype("uint64")).max
                        ):
                            partition_slices[i].append(s)
                        prev_index = partial_index
                        i += 1

            partial_write_array_ids_array = tiledb.open(
                partial_write_array_ids_uri, mode="r", timestamp=index_timestamp
            )
            partial_write_array_parts_array = tiledb.open(
                partial_write_array_parts_uri, mode="r", timestamp=index_timestamp
            )
            index_array = tiledb.open(
                index_array_uri, mode="r", timestamp=index_timestamp
            )
            ids_array = tiledb.open(ids_array_uri, mode="w", timestamp=index_timestamp)
            parts_array = tiledb.open(
                parts_array_uri, mode="w", timestamp=index_timestamp
            )
            logger.debug(
                "Partitions start: %d end: %d", partition_id_start, partition_id_end
            )
            for part in range(partition_id_start, partition_id_end, batch):
                part_end = part + batch
                if part_end > partition_id_end:
                    part_end = partition_id_end
                logger.debug(
                    "Consolidating partitions start: %d end: %d", part, part_end
                )
                read_slices = []
                for p in range(part, part_end):
                    for partition_slice in partition_slices[p]:
                        read_slices.append(partition_slice)

                start_pos = int(index_array[part]["values"])
                end_pos = int(index_array[part_end]["values"])
                if len(read_slices) == 0:
                    if start_pos != end_pos:
                        raise ValueError("Incorrect partition size.")
                    continue
                logger.debug("Read slices: %s", read_slices)
                ids = partial_write_array_ids_array.multi_index[read_slices]["values"]
                vectors = partial_write_array_parts_array.multi_index[:, read_slices][
                    "values"
                ]

                logger.debug(
                    "Ids shape %s, expected size: %d expected range:(%d,%d)",
                    ids.shape,
                    end_pos - start_pos,
                    start_pos,
                    end_pos,
                )
                if ids.shape[0] != end_pos - start_pos:
                    raise ValueError("Incorrect partition size.")

                logger.debug("Writing data to array: %s", parts_array_uri)
                parts_array[:, start_pos:end_pos] = vectors
                logger.debug("Writing data to array: %s", ids_array_uri)
                ids_array[start_pos:end_pos] = ids
            parts_array.close()
            ids_array.close()

    # --------------------------------------------------------------------
    # DAG
    # --------------------------------------------------------------------
    def submit_local(d, func, *args, **kwargs):
        # Drop kwarg
        kwargs.pop("image_name", None)
        kwargs.pop("resources", None)
        return d.submit_local(func, *args, **kwargs)

    def create_ingestion_dag(
        index_type: str,
        index_group_uri: str,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        external_ids_uri: str,
        external_ids_type: str,
        size: int,
        partitions: int,
        dimensions: int,
        copy_centroids_uri: str,
        training_sample_size: int,
        training_source_uri: Optional[str],
        training_source_type: Optional[str],
        input_vectors_per_work_item: int,
        input_vectors_work_items_per_worker: int,
        input_vectors_per_work_item_during_sampling: int,
        input_vectors_work_items_per_worker_during_sampling: int,
        table_partitions_per_work_item: int,
        table_partitions_work_items_per_worker: int,
        workers: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        use_sklearn: bool = True,
        mode: Mode = Mode.LOCAL,
        acn: Optional[str] = None,
        namespace: Optional[str] = None,
        ingest_resources: Optional[Mapping[str, Any]] = None,
        consolidate_partition_resources: Optional[Mapping[str, Any]] = None,
        copy_centroids_resources: Optional[Mapping[str, Any]] = None,
        random_sample_resources: Optional[Mapping[str, Any]] = None,
        kmeans_resources: Optional[Mapping[str, Any]] = None,
        compute_new_centroids_resources: Optional[Mapping[str, Any]] = None,
        assign_points_and_partial_new_centroids_resources: Optional[
            Mapping[str, Any]
        ] = None,
        write_centroids_resources: Optional[Mapping[str, Any]] = None,
        partial_index_resources: Optional[Mapping[str, Any]] = None,
    ) -> dag.DAG:
        kwargs = {}

        # We compute the real size of the batch in bytes.
        size_in_bytes = size * dimensions * np.dtype(vector_type).itemsize
        logger.debug("Input size in bytes: %d", size_in_bytes)
        training_sample_size_in_bytes = (
            training_sample_size * dimensions * np.dtype(vector_type).itemsize
        )
        logger.debug("Training sample size in bytes: %d", training_sample_size_in_bytes)
        if mode == Mode.BATCH:
            d = dag.DAG(
                name="vector-ingestion",
                mode=Mode.BATCH,
                max_workers=workers,
                retry_strategy=models.RetryStrategy(
                    limit=1,
                    retry_policy="Always",
                ),
                namespace=namespace,
            )
            threads = 16

            if acn:
                kwargs["access_credentials_name"] = acn
        else:
            if mode == Mode.LOCAL:
                # TODO: `default` is not an actual namespace. This is a temp fix to
                # be able to run DAGs locally.
                namespace = "default"
            d = dag.DAG(
                name="vector-ingestion",
                mode=Mode.REALTIME,
                max_workers=workers,
                namespace=namespace,
            )
            threads = multiprocessing.cpu_count()

        submit = partial(submit_local, d)
        if mode == Mode.BATCH or mode == Mode.REALTIME:
            submit = d.submit

        input_vectors_batch_size = (
            input_vectors_per_work_item * input_vectors_work_items_per_worker
        )

        # The number of vectors each task will read.
        input_vectors_batch_size_during_sampling = (
            # The number of vectors to read into memory in one batch within a task.
            input_vectors_per_work_item_during_sampling
            *
            # The number of batches that a single task will need to run.
            input_vectors_work_items_per_worker_during_sampling
        )

        def scale_resources(min_resource, max_resource, max_input_size, input_size):
            """
            Scales the resources based on the input size and the maximum input size.

            Args:
                min_resource (int): The minimum resource value (either cpu cores or ram gb).
                max_resource (int): The maximum resource value.
                max_input_size (int): The maximum input size.
                input_size (int): The input size.

            Returns:
                str: The scaled resource value as a string.
            """
            return str(
                max(
                    min_resource,
                    min(
                        max_resource,
                        int(max_resource * input_size / max_input_size),
                    ),
                )
            )

        # We can't set as default in the function due to the use of `str(threads)`
        # For consistency we then apply all defaults for resources here.
        if ingest_resources is None:
            ingest_resources = {
                "cpu": scale_resources(
                    2, threads, DEFAULT_PARTITION_BYTE_SIZE, size_in_bytes
                ),
                "memory": scale_resources(
                    2, 16, DEFAULT_PARTITION_BYTE_SIZE, size_in_bytes
                )
                + "Gi",
            }

        if consolidate_partition_resources is None:
            consolidate_partition_resources = {
                "cpu": scale_resources(
                    2, threads, DEFAULT_PARTITION_BYTE_SIZE, size_in_bytes
                ),
                "memory": scale_resources(
                    2, 16, DEFAULT_PARTITION_BYTE_SIZE, size_in_bytes
                )
                + "Gi",
            }

        if copy_centroids_resources is None:
            copy_centroids_resources = {"cpu": "1", "memory": "2Gi"}

        if random_sample_resources is None:
            random_sample_resources = {
                "cpu": "2",
                "memory": "6Gi",
            }

        if kmeans_resources is None:
            kmeans_resources = {
                "cpu": scale_resources(
                    4,
                    threads,
                    DEFAULT_KMEANS_BYTES_PER_SAMPLE,
                    training_sample_size_in_bytes,
                ),
                "memory": scale_resources(
                    8,
                    32,
                    DEFAULT_KMEANS_BYTES_PER_SAMPLE,
                    training_sample_size_in_bytes,
                )
                + "Gi",
            }

        if compute_new_centroids_resources is None:
            compute_new_centroids_resources = {
                "cpu": "1",
                "memory": "8Gi",
            }

        if assign_points_and_partial_new_centroids_resources is None:
            assign_points_and_partial_new_centroids_resources = {
                "cpu": str(threads),
                "memory": "12Gi",
            }

        if write_centroids_resources is None:
            write_centroids_resources = {"cpu": "1", "memory": "2Gi"}

        if partial_index_resources is None:
            partial_index_resources = {"cpu": "1", "memory": "2Gi"}

        if index_type == "FLAT":
            ingest_node = submit(
                ingest_flat,
                index_group_uri=index_group_uri,
                source_uri=source_uri,
                source_type=source_type,
                updates_uri=updates_uri,
                vector_type=vector_type,
                external_ids_uri=external_ids_uri,
                external_ids_type=external_ids_type,
                dimensions=dimensions,
                size=size,
                batch=input_vectors_batch_size,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
                name="ingest",
                resources=ingest_resources,
                image_name=DEFAULT_IMG_NAME,
                **kwargs,
            )
            return d
        elif is_type_erased_index(index_type):
            ingest_node = submit(
                ingest_type_erased,
                index_type=index_type,
                index_group_uri=index_group_uri,
                source_uri=source_uri,
                source_type=source_type,
                updates_uri=updates_uri,
                vector_type=vector_type,
                external_ids_uri=external_ids_uri,
                external_ids_type=external_ids_type,
                dimensions=dimensions,
                size=size,
                batch=input_vectors_batch_size,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
                name="ingest",
                resources=ingest_resources,
                image_name=DEFAULT_IMG_NAME,
                **kwargs,
            )
            return d
        elif index_type == "IVF_FLAT":
            if copy_centroids_uri is not None:
                centroids_node = submit(
                    copy_centroids,
                    index_group_uri=index_group_uri,
                    copy_centroids_uri=copy_centroids_uri,
                    partitions=partitions,
                    dimensions=dimensions,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                    name="copy-centroids",
                    resources=copy_centroids_resources,
                    image_name=DEFAULT_IMG_NAME,
                    **kwargs,
                )
            else:
                random_sample_nodes = []
                if training_sampling_policy == TrainingSamplingPolicy.RANDOM:
                    # Create an empty array to write the sampled vectors to.
                    group = tiledb.Group(index_group_uri, "w")
                    training_source_uri = create_array(
                        group=group,
                        size=training_sample_size,
                        dimensions=dimensions,
                        vector_type=vector_type,
                        array_name=TRAINING_INPUT_VECTORS_ARRAY_NAME,
                    )
                    training_source_type = "TILEDB_ARRAY"
                    group.close()

                    idx = 0
                    num_sampled = 0
                    for start in range(
                        0, size, input_vectors_batch_size_during_sampling
                    ):
                        # What vectors to read from the source_uri.
                        end = start + input_vectors_batch_size_during_sampling
                        if end > size:
                            end = size

                        # How many vectors to sample from the vectors read.
                        percent_of_data_to_read = (end - start) / size
                        num_to_sample = math.ceil(
                            training_sample_size * percent_of_data_to_read
                        )
                        if num_sampled + num_to_sample > training_sample_size:
                            num_to_sample = training_sample_size - num_sampled
                        if num_to_sample == 0:
                            continue

                        random_sample_nodes.append(
                            submit(
                                random_sample_from_input_vectors,
                                source_uri=source_uri,
                                source_type=source_type,
                                vector_type=vector_type,
                                dimensions=dimensions,
                                source_start_pos=start,
                                source_end_pos=end,
                                batch=input_vectors_per_work_item_during_sampling,
                                random_sample_size=num_to_sample,
                                output_source_uri=training_source_uri,
                                output_start_pos=num_sampled,
                                config=config,
                                verbose=verbose,
                                name="read-random-sample-" + str(idx),
                                resources=random_sample_resources,
                                image_name=DEFAULT_IMG_NAME,
                                **kwargs,
                            )
                        )
                        num_sampled += num_to_sample
                        idx += 1
                    if num_sampled != training_sample_size:
                        raise ValueError(
                            f"The random sampling ran into an issue: num_sampled ({num_sampled}) != training_sample_size ({training_sample_size})"
                        )

                if training_sample_size <= CENTRALISED_KMEANS_MAX_SAMPLE_SIZE:
                    centroids_node = submit(
                        centralised_kmeans,
                        index_group_uri=index_group_uri,
                        source_uri=source_uri,
                        source_type=source_type,
                        vector_type=vector_type,
                        partitions=partitions,
                        dimensions=dimensions,
                        training_sample_size=training_sample_size,
                        training_source_uri=training_source_uri,
                        training_source_type=training_source_type,
                        config=config,
                        verbose=verbose,
                        trace_id=trace_id,
                        use_sklearn=use_sklearn,
                        name="kmeans",
                        resources=kmeans_resources,
                        image_name=DEFAULT_IMG_NAME,
                        **kwargs,
                    )

                    for random_sample_node in random_sample_nodes:
                        centroids_node.depends_on(random_sample_node)
                else:
                    uri = (
                        training_source_uri
                        if training_source_uri is not None
                        else source_uri
                    )
                    uri_type = (
                        training_source_type
                        if training_source_uri is not None
                        else source_type
                    )
                    internal_centroids_node = submit(
                        init_centroids,
                        source_uri=uri,
                        source_type=uri_type,
                        vector_type=vector_type,
                        partitions=partitions,
                        dimensions=dimensions,
                        config=config,
                        verbose=verbose,
                        trace_id=trace_id,
                        name="init-centroids",
                        resources=copy_centroids_resources,
                        image_name=DEFAULT_IMG_NAME,
                        **kwargs,
                    )

                    for random_sample_node in random_sample_nodes:
                        internal_centroids_node.depends_on(random_sample_node)

                    for it in range(5):
                        kmeans_workers = []
                        task_id = 0
                        for i in range(
                            0, training_sample_size, input_vectors_batch_size
                        ):
                            start = i
                            end = i + input_vectors_batch_size
                            if end > size:
                                end = size
                            kmeans_workers.append(
                                submit(
                                    assign_points_and_partial_new_centroids,
                                    centroids=internal_centroids_node,
                                    source_uri=uri,
                                    source_type=uri_type,
                                    vector_type=vector_type,
                                    partitions=partitions,
                                    dimensions=dimensions,
                                    vector_start_pos=start,
                                    vector_end_pos=end,
                                    threads=threads,
                                    config=config,
                                    verbose=verbose,
                                    trace_id=trace_id,
                                    use_sklearn=use_sklearn,
                                    name="k-means-part-" + str(task_id),
                                    resources=assign_points_and_partial_new_centroids_resources,
                                    image_name=DEFAULT_IMG_NAME,
                                    **kwargs,
                                )
                            )
                            task_id += 1
                        reducers = []
                        for i in range(0, len(kmeans_workers), 10):
                            reducers.append(
                                submit(
                                    compute_new_centroids,
                                    *kmeans_workers[i : i + 10],
                                    name="update-centroids-" + str(i),
                                    resources=compute_new_centroids_resources,
                                    image_name=DEFAULT_IMG_NAME,
                                    **kwargs,
                                )
                            )
                        internal_centroids_node = submit(
                            compute_new_centroids,
                            *reducers,
                            name="update-centroids",
                            resources=compute_new_centroids_resources,
                            image_name=DEFAULT_IMG_NAME,
                            **kwargs,
                        )
                    centroids_node = submit(
                        write_centroids,
                        centroids=internal_centroids_node,
                        index_group_uri=index_group_uri,
                        partitions=partitions,
                        dimensions=dimensions,
                        config=config,
                        verbose=verbose,
                        trace_id=trace_id,
                        name="write-centroids",
                        resources=write_centroids_resources,
                        image_name=DEFAULT_IMG_NAME,
                        **kwargs,
                    )

            compute_indexes_node = submit(
                compute_partition_indexes_udf,
                index_group_uri=index_group_uri,
                partitions=partitions,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
                name="compute-indexes",
                resources=partial_index_resources,
                image_name=DEFAULT_IMG_NAME,
                **kwargs,
            )

            task_id = 0
            for i in range(0, size, input_vectors_batch_size):
                start = i
                end = i + input_vectors_batch_size
                if end > size:
                    end = size
                ingest_node = submit(
                    ingest_vectors_udf,
                    index_group_uri=index_group_uri,
                    source_uri=source_uri,
                    source_type=source_type,
                    vector_type=vector_type,
                    external_ids_uri=external_ids_uri,
                    external_ids_type=external_ids_type,
                    partitions=partitions,
                    dimensions=dimensions,
                    start=start,
                    end=end,
                    batch=input_vectors_per_work_item,
                    threads=threads,
                    updates_uri=updates_uri,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                    name="ingest-" + str(task_id),
                    resources=ingest_resources,
                    image_name=DEFAULT_IMG_NAME,
                    **kwargs,
                )
                ingest_node.depends_on(centroids_node)
                compute_indexes_node.depends_on(ingest_node)
                task_id += 1

            if updates_uri is not None:
                ingest_additions_node = submit(
                    ingest_additions_udf,
                    index_group_uri=index_group_uri,
                    updates_uri=updates_uri,
                    vector_type=vector_type,
                    write_offset=size,
                    threads=threads,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                    name="ingest-" + str(task_id),
                    resources=ingest_resources,
                    image_name=DEFAULT_IMG_NAME,
                    **kwargs,
                )
                ingest_additions_node.depends_on(centroids_node)
                compute_indexes_node.depends_on(ingest_additions_node)

            partitions_batch = (
                table_partitions_work_items_per_worker * table_partitions_per_work_item
            )
            task_id = 0
            for i in range(0, partitions, partitions_batch):
                start = i
                end = i + partitions_batch
                if end > partitions:
                    end = partitions
                consolidate_partition_node = submit(
                    consolidate_partition_udf,
                    index_group_uri=index_group_uri,
                    partition_id_start=start,
                    partition_id_end=end,
                    batch=table_partitions_per_work_item,
                    dimensions=dimensions,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                    name="consolidate-partition-" + str(task_id),
                    resources=consolidate_partition_resources,
                    image_name=DEFAULT_IMG_NAME,
                    **kwargs,
                )
                consolidate_partition_node.depends_on(compute_indexes_node)
                task_id += 1
            return d
        else:
            raise ValueError(f"Not supported index_type {index_type}")

    def consolidate_and_vacuum(
        index_group_uri: str,
        config: Optional[Mapping[str, Any]] = None,
    ):
        with tiledb.Group(index_group_uri) as group:
            write_group = tiledb.Group(index_group_uri, "w")
            try:
                if INPUT_VECTORS_ARRAY_NAME in group:
                    tiledb.Array.delete_array(group[INPUT_VECTORS_ARRAY_NAME].uri)
                    write_group.remove(INPUT_VECTORS_ARRAY_NAME)
                if EXTERNAL_IDS_ARRAY_NAME in group:
                    tiledb.Array.delete_array(group[EXTERNAL_IDS_ARRAY_NAME].uri)
                    write_group.remove(EXTERNAL_IDS_ARRAY_NAME)
            except tiledb.TileDBError as err:
                message = str(err)
                if "does not exist" not in message:
                    raise err
            write_group.close()

            modes = ["fragment_meta", "commits", "array_meta"]
            for mode in modes:
                conf = tiledb.Config(config)
                conf["sm.consolidation.mode"] = mode
                conf["sm.vacuum.mode"] = mode
                ids_uri = group[IDS_ARRAY_NAME].uri
                parts_uri = group[PARTS_ARRAY_NAME].uri
                tiledb.consolidate(parts_uri, config=conf)
                tiledb.vacuum(parts_uri, config=conf)
                tiledb.consolidate(ids_uri, config=conf)
                tiledb.vacuum(ids_uri, config=conf)
            partial_write_array_exists = PARTIAL_WRITE_ARRAY_DIR in group
        if partial_write_array_exists:
            with tiledb.Group(index_group_uri, "w") as partial_write_array_group:
                partial_write_array_group.remove(PARTIAL_WRITE_ARRAY_DIR)
            partial_write_array_dir_uri = (
                index_group_uri + "/" + PARTIAL_WRITE_ARRAY_DIR
            )
            with tiledb.Group(
                partial_write_array_dir_uri, "m"
            ) as partial_write_array_group:
                partial_write_array_group.delete(recursive=True)

    # --------------------------------------------------------------------
    # End internal function definitions
    # --------------------------------------------------------------------

    with tiledb.scope_ctx(ctx_or_config=config):
        logger = setup(config, verbose)

        if input_vectors is not None:
            in_size = input_vectors.shape[0]
            dimensions = input_vectors.shape[1]
            vector_type = input_vectors.dtype
            source_type = "TILEDB_ARRAY"
        else:
            if source_type is None:
                source_type = autodetect_source_type(source_uri=source_uri)
            in_size, dimensions, vector_type = read_source_metadata(
                source_uri=source_uri, source_type=source_type
            )

        logger.debug("Ingesting Vectors into %r", index_group_uri)
        arrays_created = False
        if is_type_erased_index(index_type):
            # If we're using a type-erased index, we create the group in C++.
            try:
                # Try opening the group to see if it exists.
                group = tiledb.Group(index_group_uri, "r")
                group.close()
                arrays_created = True
            except tiledb.TileDBError as err:
                # If it does not then we can create it in C++.
                message = str(err)
                if "not exist" in message:
                    if index_type == "VAMANA":
                        vamana_index.create(
                            uri=index_group_uri,
                            dimensions=dimensions,
                            vector_type=vector_type,
                            config=config,
                            l_build=l_build,
                            r_max_degree=r_max_degree,
                            storage_version=storage_version,
                        )
                    elif index_type == "IVF_PQ":
                        ivf_pq_index.create(
                            uri=index_group_uri,
                            dimensions=dimensions,
                            vector_type=vector_type,
                            num_subspaces=num_subspaces,
                            partitions=partitions,
                            config=config,
                            storage_version=storage_version,
                        )
                    else:
                        raise ValueError(f"Unsupported index type {index_type}")
                else:
                    raise err
        else:
            # Otherwise, we create the group in Python.
            try:
                tiledb.group_create(index_group_uri)
            except tiledb.TileDBError as err:
                message = str(err)
                if "already exists" in message:
                    arrays_created = True
                    logger.debug(f"Group '{index_group_uri}' already exists")
                else:
                    raise err
        group = tiledb.Group(index_group_uri, "r")
        ingestion_timestamps = list(
            json.loads(group.meta.get("ingestion_timestamps", "[]"))
        )
        base_sizes = list(json.loads(group.meta.get("base_sizes", "[]")))
        partition_history = list(json.loads(group.meta.get("partition_history", "[]")))
        if partitions == -1:
            partitions = int(group.meta.get("partitions", "-1"))

        previous_ingestion_timestamp = 0
        if index_timestamp is None:
            index_timestamp = int(time.time() * 1000)
        if len(ingestion_timestamps) > 0:
            previous_ingestion_timestamp = ingestion_timestamps[
                len(ingestion_timestamps) - 1
            ]
            if (
                index_timestamp is not None
                and index_timestamp <= previous_ingestion_timestamp
            ):
                raise ValueError(
                    f"New ingestion timestamp: {index_timestamp} can't be smaller that the latest ingestion "
                    f"timestamp: {previous_ingestion_timestamp}"
                )

        group.close()

        if size == -1:
            size = int(in_size)
        if size > in_size:
            size = int(in_size)
        logger.debug("Input dataset size %d", size)
        logger.debug("Input dataset dimensions %d", dimensions)
        logger.debug("Vector dimension type %s", vector_type)
        if training_sample_size > size:
            raise ValueError(
                f"training_sample_size {training_sample_size} is larger than the input dataset size {size}"
            )

        if partitions == -1:
            partitions = max(1, int(math.sqrt(size)))
        if training_sample_size == -1:
            training_sample_size = min(size, 100 * partitions)
        if mode == Mode.BATCH:
            if workers == -1:
                workers = 10
        else:
            workers = 1
        logger.debug("Partitions %d", partitions)
        logger.debug("Training sample size %d", training_sample_size)
        logger.debug(
            "Training source uri %s and type %s",
            training_source_uri,
            training_source_type,
        )
        logger.debug("Number of workers %d", workers)

        # Compute task parameters for main ingestion.
        if input_vectors_per_work_item == -1:
            # We scale the input_vectors_per_work_item to maintain the DEFAULT_PARTITION_BYTE_SIZE
            input_vectors_per_work_item = int(
                DEFAULT_PARTITION_BYTE_SIZE
                / dimensions
                / np.dtype(vector_type).itemsize
            )
        input_vectors_work_items = int(math.ceil(size / input_vectors_per_work_item))
        input_vectors_work_tasks = input_vectors_work_items
        input_vectors_work_items_per_worker = 1
        if max_tasks_per_stage == -1:
            max_tasks_per_stage = MAX_TASKS_PER_STAGE
        if input_vectors_work_tasks > max_tasks_per_stage:
            input_vectors_work_items_per_worker = int(
                math.ceil(input_vectors_work_items / max_tasks_per_stage)
            )
            input_vectors_work_tasks = max_tasks_per_stage
        logger.debug("input_vectors_per_work_item %d", input_vectors_per_work_item)
        logger.debug("input_vectors_work_items %d", input_vectors_work_items)
        logger.debug("input_vectors_work_tasks %d", input_vectors_work_tasks)
        logger.debug(
            "input_vectors_work_items_per_worker %d",
            input_vectors_work_items_per_worker,
        )

        # Compute task parameters for random sampling.
        # How many input vectors to read into memory in one batch within a task.
        if input_vectors_per_work_item_during_sampling == -1:
            input_vectors_per_work_item_during_sampling = VECTORS_PER_SAMPLE_WORK_ITEM
        # How many total batches we need to read all the data..
        input_vectors_work_items_during_sampling = int(
            math.ceil(size / input_vectors_per_work_item_during_sampling)
        )
        # The number of tasks to create, at max.
        if max_sampling_tasks == -1:
            max_sampling_tasks = MAX_TASKS_PER_STAGE
        # The number of batches a single task will run. If there are more batches required than
        # allowed tasks, each task will process mutiple batches.
        input_vectors_work_items_per_worker_during_sampling = 1
        if input_vectors_work_items_during_sampling > max_sampling_tasks:
            input_vectors_work_items_per_worker_during_sampling = int(
                math.ceil(input_vectors_work_items_during_sampling / max_sampling_tasks)
            )
            input_vectors_work_items_during_sampling = max_sampling_tasks
        logger.debug(
            "input_vectors_per_work_item_during_sampling %d",
            input_vectors_per_work_item_during_sampling,
        )
        logger.debug(
            "input_vectors_work_items_during_sampling %d",
            input_vectors_work_items_during_sampling,
        )
        logger.debug(
            "input_vectors_work_items_per_worker_during_sampling %d",
            input_vectors_work_items_per_worker_during_sampling,
        )

        vectors_per_table_partitions = max(1, size / partitions)
        table_partitions_per_work_item = max(
            1,
            int(math.ceil(input_vectors_per_work_item / vectors_per_table_partitions)),
        )
        table_partitions_work_items = int(
            math.ceil(partitions / table_partitions_per_work_item)
        )
        table_partitions_work_tasks = table_partitions_work_items
        table_partitions_work_items_per_worker = 1
        if table_partitions_work_tasks > max_tasks_per_stage:
            table_partitions_work_items_per_worker = int(
                math.ceil(table_partitions_work_items / max_tasks_per_stage)
            )
            table_partitions_work_tasks = max_tasks_per_stage
        logger.debug(
            "table_partitions_per_work_item %d", table_partitions_per_work_item
        )
        logger.debug("table_partitions_work_items %d", table_partitions_work_items)
        logger.debug("table_partitions_work_tasks %d", table_partitions_work_tasks)
        logger.debug(
            "table_partitions_work_items_per_worker %d",
            table_partitions_work_items_per_worker,
        )

        logger.debug("Creating arrays")
        group = tiledb.Group(index_group_uri, "w")
        temp_data_group = create_temp_data_group(group=group)
        create_arrays(
            group=group,
            temp_data_group=temp_data_group,
            arrays_created=arrays_created,
            index_type=index_type,
            dimensions=dimensions,
            input_vectors_work_items=input_vectors_work_items,
            vector_type=vector_type,
            logger=logger,
            storage_version=storage_version,
        )

        if training_input_vectors is not None:
            training_source_uri = write_input_vectors(
                group=temp_data_group,
                input_vectors=training_input_vectors,
                size=training_input_vectors.shape[0],
                dimensions=training_input_vectors.shape[1],
                vector_type=training_input_vectors.dtype,
                array_name=TRAINING_INPUT_VECTORS_ARRAY_NAME,
            )
            training_source_type = "TILEDB_ARRAY"

        if input_vectors is not None:
            source_uri = write_input_vectors(
                group=temp_data_group,
                input_vectors=input_vectors,
                size=in_size,
                dimensions=dimensions,
                vector_type=vector_type,
                array_name=INPUT_VECTORS_ARRAY_NAME,
            )

        if external_ids is not None:
            external_ids_uri = write_external_ids(
                group=temp_data_group,
                external_ids=external_ids,
                size=size,
                partitions=partitions,
            )
            external_ids_type = "TILEDB_ARRAY"
        else:
            if external_ids_type is None:
                external_ids_type = "U64BIN"
        temp_data_group.close()
        group.meta["temp_size"] = size
        group.close()

        logger.debug("Creating ingestion graph")
        d = create_ingestion_dag(
            index_type=index_type,
            index_group_uri=index_group_uri,
            source_uri=source_uri,
            source_type=source_type,
            vector_type=vector_type,
            external_ids_uri=external_ids_uri,
            external_ids_type=external_ids_type,
            size=size,
            partitions=partitions,
            dimensions=dimensions,
            copy_centroids_uri=copy_centroids_uri,
            training_sample_size=training_sample_size,
            training_source_uri=training_source_uri,
            training_source_type=training_source_type,
            input_vectors_per_work_item=input_vectors_per_work_item,
            input_vectors_work_items_per_worker=input_vectors_work_items_per_worker,
            input_vectors_per_work_item_during_sampling=input_vectors_per_work_item_during_sampling,
            input_vectors_work_items_per_worker_during_sampling=input_vectors_work_items_per_worker_during_sampling,
            table_partitions_per_work_item=table_partitions_per_work_item,
            table_partitions_work_items_per_worker=table_partitions_work_items_per_worker,
            workers=workers,
            config=config,
            verbose=verbose,
            trace_id=trace_id,
            use_sklearn=use_sklearn,
            mode=mode,
            acn=acn,
            namespace=namespace,
            ingest_resources=ingest_resources,
            consolidate_partition_resources=consolidate_partition_resources,
            copy_centroids_resources=copy_centroids_resources,
            random_sample_resources=random_sample_resources,
            kmeans_resources=kmeans_resources,
            compute_new_centroids_resources=compute_new_centroids_resources,
            assign_points_and_partial_new_centroids_resources=assign_points_and_partial_new_centroids_resources,
            write_centroids_resources=write_centroids_resources,
            partial_index_resources=partial_index_resources,
        )
        logger.debug("Submitting ingestion graph")
        d.compute()
        logger.debug("Submitted ingestion graph")
        d.wait()

        group = tiledb.Group(index_group_uri, "r")
        temp_size = int(group.meta.get("temp_size", "0"))
        group.close()

        if not is_type_erased_index(index_type):
            # For type-erased indexes (i.e. Vamana), we update this metadata in the write_index()
            # call during create_ingestion_dag(), so don't do it here.
            group = tiledb.Group(index_group_uri, "w")
            ingestion_timestamps.append(index_timestamp)
            base_sizes.append(temp_size)
            partition_history.append(partitions)
            group.meta["partition_history"] = json.dumps(partition_history)
            group.meta["base_sizes"] = json.dumps(base_sizes)
            group.meta["ingestion_timestamps"] = json.dumps(ingestion_timestamps)
            group.close()

        consolidate_and_vacuum(index_group_uri=index_group_uri, config=config)

        if index_type == "FLAT":
            return flat_index.FlatIndex(uri=index_group_uri, config=config)
        elif index_type == "VAMANA":
            return vamana_index.VamanaIndex(uri=index_group_uri, config=config)
        elif index_type == "IVF_FLAT":
            return ivf_flat_index.IVFFlatIndex(
                uri=index_group_uri, memory_budget=1000000, config=config
            )
        elif index_type == "IVF_PQ":
            return ivf_pq_index.IVFPQIndex(uri=index_group_uri, config=config)
        else:
            raise ValueError(f"Not supported index_type {index_type}")
