import json
from functools import partial
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from tiledb.cloud.dag import Mode

from tiledb.vector_search._tiledbvspy import *
from tiledb.vector_search.storage_formats import STORAGE_VERSION


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
    copy_centroids_uri: str = None,
    training_sample_size: int = -1,
    workers: int = -1,
    input_vectors_per_work_item: int = -1,
    max_tasks_per_stage: int= -1,
    storage_version: str = STORAGE_VERSION,
    verbose: bool = False,
    trace_id: Optional[str] = None,
    use_sklearn: bool = False,
    mode: Mode = Mode.LOCAL,
    **kwargs,
):
    """
    Ingest vectors into TileDB.

    Parameters
    ----------
    index_type: str
        Type of vector index (FLAT, IVF_FLAT)
    index_uri: str
        Vector index URI (stored as TileDB group)
    input_vectors: numpy Array
        Input vectors, if this is provided it takes precedence over source_uri and source_type.
    source_uri: str
        Data source URI
    source_type: str
        Type of the source data. If left empty it is auto-detected from the suffix of source_uri
    external_ids: numpy Array
        Input vector external_ids, if this is provided it takes precedence over external_ids_uri and external_ids_type
    external_ids_uri: str
        Source URI for external_ids
    external_ids_type: str
        File type of external_ids_uri. If left empty it is auto-detected from the suffix of external_ids_uri
    updates_uri: str
        Updates
    index_timestamp: int
        Timestamp to use for writing and reading data. By default it sues the current unix ms timestamp.
    config: None
        config dictionary, defaults to None
    namespace: str
        TileDB-Cloud namespace, defaults to None
    size: int = 1
        Number of input vectors,
        if not provided use the full size of the input dataset
    partitions: int = -1
        Number of partitions to load the data with,
        if not provided, is auto-configured based on the dataset size
    copy_centroids_uri: str
        TileDB array URI to copy centroids from,
        if not provided, centroids are build running kmeans
    training_sample_size: int = -1
        vector sample size to train centroids with,
        if not provided, is auto-configured based on the dataset sizes
    workers: int = -1
        number of workers for vector ingestion,
        if not provided, is auto-configured based on the dataset size
    input_vectors_per_work_item: int = -1
        number of vectors per ingestion work item,
        if not provided, is auto-configured
    max_tasks_per_stage: int = -1
        Max number of tasks per execution stage of ingestion,
        if not provided, is auto-configured
    storage_version: str
        Vector index storage format version.
    verbose: bool
        verbose logging, defaults to False
    trace_id: Optional[str]
        trace ID for logging, defaults to None
    use_sklearn: bool
        Whether to use scikit-learn's implementation of k-means clustering instead of
        tiledb.vector_search's. Defaults to false.
    mode: Mode
        execution mode, defaults to LOCAL use BATCH for distributed execution
    """
    import enum
    import json
    import logging
    import math
    import multiprocessing
    import os
    import time
    from datetime import datetime
    from typing import Any, Mapping

    import numpy as np
    from tiledb.cloud import dag
    from tiledb.cloud.rest_api import models
    from tiledb.cloud.utilities import get_logger, set_aws_context

    import tiledb
    from tiledb.vector_search import flat_index, ivf_flat_index
    from tiledb.vector_search.index import Index
    from tiledb.vector_search.storage_formats import storage_formats

    # use index_group_uri for internal clarity
    index_group_uri = index_uri

    CENTROIDS_ARRAY_NAME = storage_formats[storage_version]["CENTROIDS_ARRAY_NAME"]
    INDEX_ARRAY_NAME = storage_formats[storage_version]["INDEX_ARRAY_NAME"]
    IDS_ARRAY_NAME = storage_formats[storage_version]["IDS_ARRAY_NAME"]
    PARTS_ARRAY_NAME = storage_formats[storage_version]["PARTS_ARRAY_NAME"]
    INPUT_VECTORS_ARRAY_NAME = storage_formats[storage_version][
        "INPUT_VECTORS_ARRAY_NAME"
    ]
    EXTERNAL_IDS_ARRAY_NAME = storage_formats[storage_version][
        "EXTERNAL_IDS_ARRAY_NAME"
    ]
    PARTIAL_WRITE_ARRAY_DIR = storage_formats[storage_version][
        "PARTIAL_WRITE_ARRAY_DIR"
    ]
    DEFAULT_ATTR_FILTERS = storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"]
    VECTORS_PER_WORK_ITEM = 20000000
    MAX_TASKS_PER_STAGE = 100
    CENTRALISED_KMEANS_MAX_SAMPLE_SIZE = 1000000
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
            return size, dimensions, schema.attr("values").dtype
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
            raise ValueError(f"Not supported source_type {source_type}")

    def write_input_vectors(
        group: tiledb.Group,
        input_vectors: np.ndarray,
        size: int,
        dimensions: int,
        vector_type: np.dtype,
    ) -> str:
        input_vectors_array_uri = f"{group.uri}/{INPUT_VECTORS_ARRAY_NAME}"
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
        group.add(input_vectors_array_uri, name=INPUT_VECTORS_ARRAY_NAME)

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
        group.add(external_ids_array_uri, name=IDS_ARRAY_NAME)

        external_ids_array = tiledb.open(
            external_ids_array_uri, "w", timestamp=index_timestamp
        )
        external_ids_array[:] = external_ids
        external_ids_array.close()

        return external_ids_array_uri

    def create_arrays(
        group: tiledb.Group,
        arrays_created: bool,
        index_type: str,
        size: int,
        dimensions: int,
        input_vectors_work_items: int,
        vector_type: np.dtype,
        logger: logging.Logger,
    ) -> None:
        if index_type == "FLAT":
            if not arrays_created:
                flat_index.create(
                    uri=group.uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    group_exists=True,
                    config=config,
                )
        elif index_type == "IVF_FLAT":
            if not arrays_created:
                ivf_flat_index.create(
                    uri=group.uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    group_exists=True,
                    config=config,
                )
            tile_size = int(
                ivf_flat_index.TILE_SIZE_BYTES
                / np.dtype(vector_type).itemsize
                / dimensions
            )
            partial_write_array_dir_uri = f"{group.uri}/{PARTIAL_WRITE_ARRAY_DIR}"
            partial_write_array_index_uri = (
                f"{partial_write_array_dir_uri}/{INDEX_ARRAY_NAME}"
            )
            partial_write_array_ids_uri = (
                f"{partial_write_array_dir_uri}/{IDS_ARRAY_NAME}"
            )
            partial_write_array_parts_uri = (
                f"{partial_write_array_dir_uri}/{PARTS_ARRAY_NAME}"
            )

            try:
                tiledb.group_create(partial_write_array_dir_uri)
            except tiledb.TileDBError as err:
                message = str(err)
                if "already exists" in message:
                    logger.debug(
                        f"Group '{partial_write_array_dir_uri}' already exists"
                    )
                raise err
            partial_write_array_group = tiledb.Group(partial_write_array_dir_uri, "w")
            group.add(partial_write_array_dir_uri, name=PARTIAL_WRITE_ARRAY_DIR)

            try:
                tiledb.group_create(partial_write_array_index_uri)
            except tiledb.TileDBError as err:
                message = str(err)
                if "already exists" in message:
                    logger.debug(
                        f"Group '{partial_write_array_index_uri}' already exists"
                    )
                raise err
            partial_write_array_group.add(
                partial_write_array_index_uri, name=INDEX_ARRAY_NAME
            )
            partial_write_array_index_group = tiledb.Group(
                partial_write_array_index_uri, "w"
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
                    filters=DEFAULT_ATTR_FILTERS,
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
                partial_write_array_group.add(
                    partial_write_array_ids_uri, name=IDS_ARRAY_NAME
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
                parts_array_dom = tiledb.Domain(
                    parts_array_rows_dim, parts_array_cols_dim
                )
                parts_attr = tiledb.Attr(
                    name="values", dtype=vector_type, filters=DEFAULT_ATTR_FILTERS
                )
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
                partial_write_array_group.add(
                    partial_write_array_parts_uri, name=PARTS_ARRAY_NAME
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
                    partial_write_array_index_group.add(part_index_uri, name=str(part))
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
                    partial_write_array_index_group.add(part_index_uri, name="additions")
            partial_write_array_group.close()
            partial_write_array_index_group.close()

        else:
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
                return np.transpose(
                    src_array[0:dimensions, start_pos:end_pos]["values"]
                ).copy(order="C")
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
        src_centroids = src[:, :]
        dest[:, :] = src_centroids
        logger.debug(src_centroids)

    # --------------------------------------------------------------------
    # centralised kmeans UDFs
    # --------------------------------------------------------------------
    def centralised_kmeans(
        index_group_uri: str,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        partitions: int,
        dimensions: int,
        sample_start_pos: int,
        sample_end_pos: int,
        init: str = "random",
        max_iter: int = 10,
        n_init: int = 1,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        use_sklearn: bool = False
    ):
        from sklearn.cluster import KMeans

        from tiledb.vector_search.module import (
            array_to_matrix,
            kmeans_fit,
        )
        with tiledb.scope_ctx(ctx_or_config=config):
            logger = setup(config, verbose)
            group = tiledb.Group(index_group_uri)
            centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
            verb = 0
            if verbose:
                verb = 3

            if sample_end_pos - sample_start_pos >= partitions:
                sample_vectors = read_input_vectors(
                    source_uri=source_uri,
                    source_type=source_type,
                    vector_type=vector_type,
                    dimensions=dimensions,
                    start_pos=sample_start_pos,
                    end_pos=sample_end_pos,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                ).astype(np.float32)

                if use_sklearn:
                    km = KMeans(
                        n_clusters=partitions,
                        init=init,
                        max_iter=max_iter,
                        verbose=verb,
                        n_init=n_init,
                    )
                    km.fit_predict(sample_vectors)
                    centroids = np.transpose(np.array(km.cluster_centers_))
                else:
                    centroids = kmeans_fit(partitions, init, max_iter, verbose, n_init, array_to_matrix(np.transpose(sample_vectors)))
                    centroids = np.array(centroids) # TODO: why is this here?
            else:
                centroids = np.random.rand(dimensions, partitions)

            logger.debug("Start kmeans training")

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
        use_sklearn: bool = False,
    ):
        import tiledb.cloud
        from sklearn.cluster import KMeans

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
                worker = mp.Process(
                    target=generate_new_centroid_per_thread,
                    args=(
                        thread_id,
                        i,
                        i + batch_size,
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
                km.n_threads_ = threads
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

        from tiledb.vector_search.module import (StdVector_u64,
                                                 array_to_matrix, ivf_index,
                                                 ivf_index_tdb)

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

            part_name = str(part) + "-" + str(part_end)

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

        from tiledb.vector_search.module import (StdVector_u64,
                                                 array_to_matrix, ivf_index)

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
        partial_write_array_index_uri = partial_write_array_index_group[
            "additions"
        ].uri
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
        input_vectors_per_work_item: int,
        input_vectors_work_items_per_worker: int,
        table_partitions_per_work_item: int,
        table_partitions_work_items_per_worker: int,
        workers: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        use_sklearn: bool = False,
        mode: Mode = Mode.LOCAL,
    ) -> dag.DAG:
        if mode == Mode.BATCH:
            d = dag.DAG(
                name="vector-ingestion",
                mode=Mode.BATCH,
                max_workers=workers,
                retry_strategy=models.RetryStrategy(
                    limit=1,
                    retry_policy="Always",
                ),
            )
            threads = 16
        else:
            d = dag.DAG(
                name="vector-ingestion",
                mode=Mode.REALTIME,
                max_workers=workers,
                namespace="default",
            )
            threads = multiprocessing.cpu_count()

        submit = partial(submit_local, d)
        if mode == Mode.BATCH or mode == Mode.REALTIME:
            submit = d.submit

        input_vectors_batch_size = (
            input_vectors_per_work_item * input_vectors_work_items_per_worker
        )
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
                resources={"cpu": str(threads), "memory": "16Gi"},
                image_name=DEFAULT_IMG_NAME,
            )
            return d
        elif index_type == "IVF_FLAT":
            if copy_centroids_uri is not None:
                centroids_node = submit(
                    copy_centroids,
                    index_group_uri=index_group_uri,
                    copy_centroids_uri=copy_centroids_uri,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                    name="copy-centroids",
                    resources={"cpu": "1", "memory": "2Gi"},
                    image_name=DEFAULT_IMG_NAME,
                )
            else:
                if training_sample_size <= CENTRALISED_KMEANS_MAX_SAMPLE_SIZE:
                    centroids_node = submit(
                        centralised_kmeans,
                        index_group_uri=index_group_uri,
                        source_uri=source_uri,
                        source_type=source_type,
                        vector_type=vector_type,
                        partitions=partitions,
                        dimensions=dimensions,
                        sample_start_pos=0,
                        sample_end_pos=training_sample_size,
                        config=config,
                        verbose=verbose,
                        trace_id=trace_id,
                        use_sklearn=use_sklearn,
                        name="kmeans",
                        resources={"cpu": "8", "memory": "32Gi"},
                        image_name=DEFAULT_IMG_NAME,
                    )
                else:
                    internal_centroids_node = submit(
                        init_centroids,
                        source_uri=source_uri,
                        source_type=source_type,
                        vector_type=vector_type,
                        partitions=partitions,
                        dimensions=dimensions,
                        config=config,
                        verbose=verbose,
                        trace_id=trace_id,
                        name="init-centroids",
                        resources={"cpu": "1", "memory": "1Gi"},
                        image_name=DEFAULT_IMG_NAME,
                    )

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
                                    source_uri=source_uri,
                                    source_type=source_type,
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
                                    resources={"cpu": str(threads), "memory": "12Gi"},
                                    image_name=DEFAULT_IMG_NAME,
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
                                    resources={"cpu": "1", "memory": "8Gi"},
                                    image_name=DEFAULT_IMG_NAME,
                                )
                            )
                        internal_centroids_node = submit(
                            compute_new_centroids,
                            *reducers,
                            name="update-centroids",
                            resources={"cpu": "1", "memory": "8Gi"},
                            image_name=DEFAULT_IMG_NAME,
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
                        resources={"cpu": "1", "memory": "2Gi"},
                        image_name=DEFAULT_IMG_NAME,
                    )

            compute_indexes_node = submit(
                compute_partition_indexes_udf,
                index_group_uri=index_group_uri,
                partitions=partitions,
                config=config,
                verbose=verbose,
                trace_id=trace_id,
                name="compute-indexes",
                resources={"cpu": "1", "memory": "2Gi"},
                image_name=DEFAULT_IMG_NAME,
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
                    resources={"cpu": str(threads), "memory": "16Gi"},
                    image_name=DEFAULT_IMG_NAME,
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
                    resources={"cpu": str(threads), "memory": "16Gi"},
                    image_name=DEFAULT_IMG_NAME,
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
                    resources={"cpu": str(threads), "memory": "16Gi"},
                    image_name=DEFAULT_IMG_NAME,
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
        group = tiledb.Group(index_group_uri)
        try:
            if INPUT_VECTORS_ARRAY_NAME in group:
                tiledb.Array.delete_array(group[INPUT_VECTORS_ARRAY_NAME].uri)
            if EXTERNAL_IDS_ARRAY_NAME in group:
                tiledb.Array.delete_array(group[EXTERNAL_IDS_ARRAY_NAME].uri)
        except tiledb.TileDBError as err:
            message = str(err)
            if "does not exist" not in message:
                raise err
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
        group.close()

        # TODO remove temp data for tiledb URIs
        if not index_group_uri.startswith("tiledb://"):
            group = tiledb.Group(index_group_uri, "r")
            if PARTIAL_WRITE_ARRAY_DIR in group:
                group.close()
                group = tiledb.Group(index_group_uri, "w")
                group.remove(PARTIAL_WRITE_ARRAY_DIR)
                vfs = tiledb.VFS(config)
                partial_write_array_dir_uri = (
                    index_group_uri + "/" + PARTIAL_WRITE_ARRAY_DIR
                )
                if vfs.is_dir(partial_write_array_dir_uri):
                    vfs.remove_dir(partial_write_array_dir_uri)
            group.close()

    # --------------------------------------------------------------------
    # End internal function definitions
    # --------------------------------------------------------------------

    with tiledb.scope_ctx(ctx_or_config=config):
        logger = setup(config, verbose)
        logger.debug("Ingesting Vectors into %r", index_group_uri)
        arrays_created = False
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
        group = tiledb.Group(index_group_uri, "w")

        if input_vectors is not None:
            in_size = input_vectors.shape[0]
            dimensions = input_vectors.shape[1]
            vector_type = input_vectors.dtype
            source_uri = write_input_vectors(
                group=group,
                input_vectors=input_vectors,
                size=in_size,
                dimensions=dimensions,
                vector_type=vector_type,
            )
            source_type = "TILEDB_ARRAY"
        else:
            if source_type is None:
                source_type = autodetect_source_type(source_uri=source_uri)
            in_size, dimensions, vector_type = read_source_metadata(
                source_uri=source_uri, source_type=source_type
            )
        if size == -1:
            size = int(in_size)
        if size > in_size:
            size = int(in_size)
        logger.debug("Input dataset size %d", size)
        logger.debug("Input dataset dimensions %d", dimensions)
        logger.debug("Vector dimension type %s", vector_type)
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
        logger.debug("Number of workers %d", workers)

        if external_ids is not None:
            external_ids_uri = write_external_ids(
                group=group,
                external_ids=external_ids,
                size=in_size,
                partitions=partitions,
            )
            external_ids_type = "TILEDB_ARRAY"
        else:
            if external_ids_type is None:
                external_ids_type = "U64BIN"

        if input_vectors_per_work_item == -1:
            input_vectors_per_work_item = VECTORS_PER_WORK_ITEM
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
        create_arrays(
            group=group,
            arrays_created=arrays_created,
            index_type=index_type,
            size=size,
            dimensions=dimensions,
            input_vectors_work_items=input_vectors_work_items,
            vector_type=vector_type,
            logger=logger,
        )
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
            input_vectors_per_work_item=input_vectors_per_work_item,
            input_vectors_work_items_per_worker=input_vectors_work_items_per_worker,
            table_partitions_per_work_item=table_partitions_per_work_item,
            table_partitions_work_items_per_worker=table_partitions_work_items_per_worker,
            workers=workers,
            config=config,
            verbose=verbose,
            trace_id=trace_id,
            use_sklearn=use_sklearn,
            mode=mode,
        )
        logger.debug("Submitting ingestion graph")
        d.compute()
        logger.debug("Submitted ingestion graph")
        d.wait()

        group = tiledb.Group(index_group_uri, "r")
        temp_size = int(group.meta.get("temp_size", "0"))
        group.close()
        group = tiledb.Group(index_group_uri, "w")
        if index_timestamp is None:
            index_timestamp = int(time.time() * 1000)
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
        elif index_type == "IVF_FLAT":
            return ivf_flat_index.IVFFlatIndex(
                uri=index_group_uri, memory_budget=1000000, config=config
            )
