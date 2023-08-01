from typing import Optional, Tuple
from functools import partial

from tiledb.cloud.dag import Mode
from tiledb.vector_search.index import FlatIndex, IVFFlatIndex, Index


def ingest(
    index_type: str,
    array_uri: str,
    source_uri: str,
    source_type: str,
    *,
    config=None,
    namespace: Optional[str] = None,
    size: int = -1,
    partitions: int = -1,
    copy_centroids_uri: str = None,
    training_sample_size: int = -1,
    workers: int = -1,
    input_vectors_per_work_item: int = -1,
    verbose: bool = False,
    trace_id: Optional[str] = None,
    mode: Mode = Mode.LOCAL,
) -> Index:
    """
    Ingest vectors into TileDB.

    Parameters
    ----------
    index_type: str
        Type of vector index (FLAT, IVF_FLAT)
    array_uri: str
        Vector array URI
    source_uri: str
        Data source URI
    source_type: str
        Type of the source data
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
        if not provided, is auto-configured based on the dataset size
    workers: int = -1
        number of workers for vector ingestion,
        if not provided, is auto-configured based on the dataset size
    input_vectors_per_work_item: int = -1
        number of vectors per ingestion work item,
        if not provided, is auto-configured
    verbose: bool
        verbose logging, defaults to False
    trace_id: Optional[str]
        trace ID for logging, defaults to None
    mode: Mode
        execution mode, defaults to LOCAL use BATCH for distributed execution
    """
    import enum
    import logging
    import math
    from typing import Any, Mapping
    import multiprocessing
    import os

    import numpy as np

    import tiledb
    from tiledb.cloud import dag
    from tiledb.cloud.rest_api import models
    from tiledb.cloud.utilities import get_logger
    from tiledb.cloud.utilities import set_aws_context

    CENTROIDS_ARRAY_NAME = "centroids.tdb"
    INDEX_ARRAY_NAME = "index.tdb"
    IDS_ARRAY_NAME = "ids.tdb"
    PARTS_ARRAY_NAME = "parts.tdb"
    PARTIAL_WRITE_ARRAY_DIR = "write_temp"
    VECTORS_PER_WORK_ITEM = 20000000
    MAX_TASKS_PER_STAGE = 100
    CENTRALISED_KMEANS_MAX_SAMPLE_SIZE = 1000000
    DEFAULT_IMG_NAME = "3.9-vectorsearch"

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

    def read_source_metadata(
        source_uri: str, source_type: str, logger: logging.Logger
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

    def create_arrays(
        group: tiledb.Group,
        index_type: str,
        size: int,
        dimensions: int,
        partitions: int,
        input_vectors_work_tasks: int,
        vector_type: np.dtype,
        logger: logging.Logger,
    ) -> None:
        if index_type == "FLAT":
            parts_uri = f"{group.uri}/{PARTS_ARRAY_NAME}"
            if not tiledb.array_exists(parts_uri):
                logger.debug("Creating parts array")
                parts_array_rows_dim = tiledb.Dim(
                    name="rows",
                    domain=(0, dimensions - 1),
                    tile=dimensions,
                    dtype=np.dtype(np.int32),
                )
                parts_array_cols_dim = tiledb.Dim(
                    name="cols",
                    domain=(0, size - 1),
                    tile=int(size / partitions),
                    dtype=np.dtype(np.int32),
                )
                parts_array_dom = tiledb.Domain(
                    parts_array_rows_dim, parts_array_cols_dim
                )
                parts_attr = tiledb.Attr(name="values", dtype=vector_type)
                parts_schema = tiledb.ArraySchema(
                    domain=parts_array_dom,
                    sparse=False,
                    attrs=[parts_attr],
                    capacity=int(size / partitions) * dimensions,
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.debug(parts_schema)
                tiledb.Array.create(parts_uri, parts_schema)
                group.add(parts_uri, name=PARTS_ARRAY_NAME)

        elif index_type == "IVF_FLAT":
            centroids_uri = f"{group.uri}/{CENTROIDS_ARRAY_NAME}"
            index_uri = f"{group.uri}/{INDEX_ARRAY_NAME}"
            ids_uri = f"{group.uri}/{IDS_ARRAY_NAME}"
            parts_uri = f"{group.uri}/{PARTS_ARRAY_NAME}"
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

            if not tiledb.array_exists(centroids_uri):
                logger.debug("Creating centroids array")
                centroids_array_rows_dim = tiledb.Dim(
                    name="rows",
                    domain=(0, dimensions - 1),
                    tile=dimensions,
                    dtype=np.dtype(np.int32),
                )
                centroids_array_cols_dim = tiledb.Dim(
                    name="cols",
                    domain=(0, partitions - 1),
                    tile=partitions,
                    dtype=np.dtype(np.int32),
                )
                centroids_array_dom = tiledb.Domain(
                    centroids_array_rows_dim, centroids_array_cols_dim
                )
                centroids_attr = tiledb.Attr(
                    name="centroids", dtype=np.dtype(np.float32)
                )
                centroids_schema = tiledb.ArraySchema(
                    domain=centroids_array_dom,
                    sparse=False,
                    attrs=[centroids_attr],
                    capacity=dimensions * partitions,
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.debug(centroids_schema)
                tiledb.Array.create(centroids_uri, centroids_schema)
                group.add(centroids_uri, name=CENTROIDS_ARRAY_NAME)

            if not tiledb.array_exists(index_uri):
                logger.debug("Creating index array")
                index_array_rows_dim = tiledb.Dim(
                    name="rows",
                    domain=(0, partitions),
                    tile=partitions,
                    dtype=np.dtype(np.int32),
                )
                index_array_dom = tiledb.Domain(index_array_rows_dim)
                index_attr = tiledb.Attr(name="values", dtype=np.dtype(np.uint64))
                index_schema = tiledb.ArraySchema(
                    domain=index_array_dom,
                    sparse=False,
                    attrs=[index_attr],
                    capacity=partitions,
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.debug(index_schema)
                tiledb.Array.create(index_uri, index_schema)
                group.add(index_uri, name=INDEX_ARRAY_NAME)

            if not tiledb.array_exists(ids_uri):
                logger.debug("Creating ids array")
                ids_array_rows_dim = tiledb.Dim(
                    name="rows",
                    domain=(0, size - 1),
                    tile=int(size / partitions),
                    dtype=np.dtype(np.int32),
                )
                ids_array_dom = tiledb.Domain(ids_array_rows_dim)
                ids_attr = tiledb.Attr(name="values", dtype=np.dtype(np.uint64))
                ids_schema = tiledb.ArraySchema(
                    domain=ids_array_dom,
                    sparse=False,
                    attrs=[ids_attr],
                    capacity=int(size / partitions),
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.debug(ids_schema)
                tiledb.Array.create(ids_uri, ids_schema)
                group.add(ids_uri, name=IDS_ARRAY_NAME)

            if not tiledb.array_exists(parts_uri):
                logger.debug("Creating parts array")
                parts_array_rows_dim = tiledb.Dim(
                    name="rows",
                    domain=(0, dimensions - 1),
                    tile=dimensions,
                    dtype=np.dtype(np.int32),
                )
                parts_array_cols_dim = tiledb.Dim(
                    name="cols",
                    domain=(0, size - 1),
                    tile=int(size / partitions),
                    dtype=np.dtype(np.int32),
                )
                parts_array_dom = tiledb.Domain(
                    parts_array_rows_dim, parts_array_cols_dim
                )
                parts_attr = tiledb.Attr(name="values", dtype=vector_type)
                parts_schema = tiledb.ArraySchema(
                    domain=parts_array_dom,
                    sparse=False,
                    attrs=[parts_attr],
                    capacity=int(size / partitions) * dimensions,
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.debug(parts_schema)
                tiledb.Array.create(parts_uri, parts_schema)
                group.add(parts_uri, name=PARTS_ARRAY_NAME)

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
                    domain=(0, size - 1),
                    tile=int(size / partitions),
                    dtype=np.dtype(np.int32),
                )
                ids_array_dom = tiledb.Domain(ids_array_rows_dim)
                ids_attr = tiledb.Attr(name="values", dtype=np.dtype(np.uint64))
                ids_schema = tiledb.ArraySchema(
                    domain=ids_array_dom,
                    sparse=False,
                    attrs=[ids_attr],
                    capacity=int(size / partitions),
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
                    domain=(0, size - 1),
                    tile=int(size / partitions),
                    dtype=np.dtype(np.int32),
                )
                parts_array_dom = tiledb.Domain(
                    parts_array_rows_dim, parts_array_cols_dim
                )
                parts_attr = tiledb.Attr(name="values", dtype=vector_type)
                parts_schema = tiledb.ArraySchema(
                    domain=parts_array_dom,
                    sparse=False,
                    attrs=[parts_attr],
                    capacity=int(size / partitions) * dimensions,
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.debug(parts_schema)
                logger.debug(partial_write_array_parts_uri)
                tiledb.Array.create(partial_write_array_parts_uri, parts_schema)
                partial_write_array_group.add(
                    partial_write_array_parts_uri, name=PARTS_ARRAY_NAME
                )

            for part in range(input_vectors_work_tasks):
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
                    index_attr = tiledb.Attr(name="values", dtype=np.dtype(np.uint64))
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
            partial_write_array_group.close()
            partial_write_array_index_group.close()

        else:
            raise ValueError(f"Not supported index_type {index_type}")

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
    ) -> np.array:
        logger = setup(config, verbose)
        logger.debug(
            "Reading input vectors start_pos: %i, end_pos: %i", start_pos, end_pos
        )
        if source_type == "TILEDB_ARRAY":
            with tiledb.open(source_uri, mode="r") as src_array:
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
        array_uri: str,
        copy_centroids_uri: str,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        logger = setup(config, verbose)
        group = tiledb.Group(array_uri)
        centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
        logger.debug(
            "Copying centroids from: %s, to: %s", copy_centroids_uri, centroids_uri
        )
        src = tiledb.open(copy_centroids_uri, mode="r")
        dest = tiledb.open(centroids_uri, mode="w")
        src_centroids = src[:, :]
        dest[:, :] = src_centroids
        logger.debug(src_centroids)

    # --------------------------------------------------------------------
    # centralised kmeans UDFs
    # --------------------------------------------------------------------
    def centralised_kmeans(
        array_uri: str,
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
    ):
        from sklearn.cluster import KMeans

        with tiledb.scope_ctx(ctx_or_config=config):
            logger = setup(config, verbose)
            group = tiledb.Group(array_uri)
            centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
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
            verb = 0
            if verbose:
                verb = 3
            logger.debug("Start kmeans training")
            km = KMeans(
                n_clusters=partitions,
                init=init,
                max_iter=max_iter,
                verbose=verb,
                n_init=n_init,
            )
            km.fit_predict(sample_vectors)
            logger.debug("Writing centroids to array %s", centroids_uri)
            with tiledb.open(centroids_uri, mode="w") as A:
                A[0:dimensions, 0:partitions] = np.transpose(
                    np.array(km.cluster_centers_)
                )

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
    ) -> np.array:
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
        centroids: np.array,
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
            km = KMeans()
            km._n_threads = threads
            km.cluster_centers_ = centroids
            assignments = km.predict(vectors)
            logger.debug("Assignments: %s", assignments[0:100])
            partial_new_centroids = update_centroids()
            logger.debug("New centroids: %s", partial_new_centroids[0:5])
            return partial_new_centroids

    def compute_new_centroids(*argv):
        import numpy as np

        return np.mean(argv, axis=0).astype(np.float32)

    def ingest_flat(
        array_uri: str,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        dimensions: int,
        start: int,
        end: int,
        batch: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        import numpy as np

        import tiledb.cloud

        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(array_uri)
            parts_array_uri = group[PARTS_ARRAY_NAME].uri
            target = tiledb.open(parts_array_uri, mode="w")
            logger.debug("Input vectors start_pos: %d, end_pos: %d", start, end)

            for part in range(start, end, batch):
                part_end = part + batch
                if part_end > end:
                    part_end = end
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

                logger.debug("Vector read: %d", len(in_vectors))
                logger.debug("Writing data to array %s", parts_array_uri)
                target[0:dimensions, start:end] = np.transpose(in_vectors)
            target.close()

    def write_centroids(
        centroids: np.array,
        array_uri: str,
        partitions: int,
        dimensions: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        with tiledb.scope_ctx(ctx_or_config=config):
            logger = setup(config, verbose)
            group = tiledb.Group(array_uri)
            centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
            logger.debug("Writing centroids to array %s", centroids_uri)
            with tiledb.open(centroids_uri, mode="w") as A:
                A[0:dimensions, 0:partitions] = np.transpose(np.array(centroids))

    # --------------------------------------------------------------------
    # vector ingestion UDFs
    # --------------------------------------------------------------------
    def ingest_vectors_udf(
        array_uri: str,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        partitions: int,
        dimensions: int,
        start: int,
        end: int,
        batch: int,
        threads: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        import tiledb.cloud
        from tiledb.vector_search.module import (
            ivf_index_tdb,
            ivf_index,
            array_to_matrix,
        )

        logger = setup(config, verbose)
        group = tiledb.Group(array_uri)
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
                str(int(start / batch))
            ].uri
            logger.debug("Input vectors start_pos: %d, end_pos: %d", part, part_end)
            if source_type == "TILEDB_ARRAY":
                logger.debug("Start indexing")
                ivf_index_tdb(
                    dtype=vector_type,
                    db_uri=source_uri,
                    centroids_uri=centroids_uri,
                    parts_uri=partial_write_array_parts_uri,
                    index_uri=partial_write_array_index_uri,
                    id_uri=partial_write_array_ids_uri,
                    start=part,
                    end=part_end,
                    nthreads=threads,
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
                logger.debug("Start indexing")
                ivf_index(
                    dtype=vector_type,
                    db=array_to_matrix(np.transpose(in_vectors).astype(vector_type)),
                    centroids_uri=centroids_uri,
                    parts_uri=partial_write_array_parts_uri,
                    index_uri=partial_write_array_index_uri,
                    id_uri=partial_write_array_ids_uri,
                    start=part,
                    end=part_end,
                    nthreads=threads,
                    config=config,
                )

    def compute_partition_indexes_udf(
        array_uri: str,
        partitions: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(array_uri)
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
                    partial_index_array = tiledb.open(partial_index_array_uri, mode="r")
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
            logger.debug(f"Partition indexes: {indexes}")
            index_array = tiledb.open(index_array_uri, mode="w")
            index_array[:] = indexes

    def consolidate_partition_udf(
        array_uri: str,
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
            group = tiledb.Group(array_uri)
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
            vfs = tiledb.VFS()
            partition_slices = []
            for i in range(partitions):
                partition_slices.append([])
            for part in partial_write_array_index_group:
                partial_index_array_uri = part.uri
                if tiledb.array_exists(partial_index_array_uri):
                    partial_index_array = tiledb.open(partial_index_array_uri, mode="r")
                    partial_indexes = partial_index_array[:]["values"]
                    prev_index = partial_indexes[0]
                    i = 0
                    for partial_index in partial_indexes[1:]:
                        s = slice(int(prev_index), int(partial_index - 1))
                        if s.start <= s.stop:
                            partition_slices[i].append(s)
                        prev_index = partial_index
                        i += 1

            partial_write_array_ids_array = tiledb.open(
                partial_write_array_ids_uri, mode="r"
            )
            partial_write_array_parts_array = tiledb.open(
                partial_write_array_parts_uri, mode="r"
            )
            index_array = tiledb.open(index_array_uri, mode="r")
            ids_array = tiledb.open(ids_array_uri, mode="w")
            parts_array = tiledb.open(parts_array_uri, mode="w")
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
        array_uri: str,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
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
            task_id = 0
            for i in range(0, size, input_vectors_batch_size):
                start = i
                end = i + input_vectors_batch_size
                if end > size:
                    end = size
                ingest_node = submit(
                    ingest_flat,
                    array_uri=array_uri,
                    source_uri=source_uri,
                    source_type=source_type,
                    vector_type=vector_type,
                    dimensions=dimensions,
                    start=start,
                    end=end,
                    batch=input_vectors_batch_size,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                    name="ingest-" + str(task_id),
                    resources={"cpu": str(threads), "memory": "16Gi"},
                    image_name=DEFAULT_IMG_NAME,
                )
                task_id += 1
            return d
        elif index_type == "IVF_FLAT":
            if copy_centroids_uri is not None:
                centroids_node = submit(
                    copy_centroids,
                    array_uri=array_uri,
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
                        array_uri=array_uri,
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
                        array_uri=array_uri,
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
                array_uri=array_uri,
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
                    array_uri=array_uri,
                    source_uri=source_uri,
                    source_type=source_type,
                    vector_type=vector_type,
                    partitions=partitions,
                    dimensions=dimensions,
                    start=start,
                    end=end,
                    batch=input_vectors_per_work_item,
                    threads=threads,
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
                    array_uri=array_uri,
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
        array_uri: str,
        config: Optional[Mapping[str, Any]] = None,
    ):
        modes = ["fragment_meta", "commits", "array_meta"]
        for mode in modes:
            conf = tiledb.Config(config)
            conf["sm.consolidation.mode"] = mode
            conf["sm.vacuum.mode"] = mode
            group = tiledb.Group(array_uri, config=conf)
            tiledb.consolidate(group[PARTS_ARRAY_NAME].uri, config=conf)
            tiledb.vacuum(group[PARTS_ARRAY_NAME].uri, config=conf)
            if index_type == "IVF_FLAT":
                tiledb.consolidate(group[IDS_ARRAY_NAME].uri, config=conf)
                tiledb.vacuum(group[IDS_ARRAY_NAME].uri, config=conf)

        # TODO remove temp data for tiledb URIs
        if not array_uri.startswith("tiledb://"):
            vfs = tiledb.VFS(config)
            partial_write_array_dir_uri = array_uri + "/" + PARTIAL_WRITE_ARRAY_DIR
            if vfs.is_dir(partial_write_array_dir_uri):
                vfs.remove_dir(partial_write_array_dir_uri)

    with tiledb.scope_ctx(ctx_or_config=config):
        logger = setup(config, verbose)
        logger.debug("Ingesting Vectors into %r", array_uri)
        try:
            tiledb.group_create(array_uri)
        except tiledb.TileDBError as err:
            message = str(err)
            if "already exists" in message:
                logger.debug(f"Group '{array_uri}' already exists")
            raise err
        group = tiledb.Group(array_uri, "w")
        group.meta["dataset_type"] = "vector_search"

        in_size, dimensions, vector_type = read_source_metadata(
            source_uri=source_uri, source_type=source_type, logger=logger
        )
        if size == -1:
            size = in_size
        if size > in_size:
            size = in_size
        logger.debug("Input dataset size %d", size)
        logger.debug("Input dataset dimensions %d", dimensions)
        logger.debug("Vector dimension type %s", vector_type)
        if partitions == -1:
            partitions = int(math.sqrt(size))
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

        if input_vectors_per_work_item == -1:
            input_vectors_per_work_item = VECTORS_PER_WORK_ITEM
        input_vectors_work_items = int(math.ceil(size / input_vectors_per_work_item))
        input_vectors_work_tasks = input_vectors_work_items
        input_vectors_work_items_per_worker = 1
        if input_vectors_work_tasks > MAX_TASKS_PER_STAGE:
            input_vectors_work_items_per_worker = int(
                math.ceil(input_vectors_work_items / MAX_TASKS_PER_STAGE)
            )
            input_vectors_work_tasks = MAX_TASKS_PER_STAGE
        logger.debug("input_vectors_per_work_item %d", input_vectors_per_work_item)
        logger.debug("input_vectors_work_items %d", input_vectors_work_items)
        logger.debug("input_vectors_work_tasks %d", input_vectors_work_tasks)
        logger.debug(
            "input_vectors_work_items_per_worker %d",
            input_vectors_work_items_per_worker,
        )

        vectors_per_table_partitions = size / partitions
        table_partitions_per_work_item = int(
            math.ceil(input_vectors_per_work_item / vectors_per_table_partitions)
        )
        table_partitions_work_items = int(
            math.ceil(partitions / table_partitions_per_work_item)
        )
        table_partitions_work_tasks = table_partitions_work_items
        table_partitions_work_items_per_worker = 1
        if table_partitions_work_tasks > MAX_TASKS_PER_STAGE:
            table_partitions_work_items_per_worker = int(
                math.ceil(table_partitions_work_items / MAX_TASKS_PER_STAGE)
            )
            table_partitions_work_tasks = MAX_TASKS_PER_STAGE
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
            index_type=index_type,
            size=size,
            dimensions=dimensions,
            partitions=partitions,
            input_vectors_work_tasks=input_vectors_work_tasks,
            vector_type=vector_type,
            logger=logger,
        )
        group.close()

        logger.debug("Creating ingestion graph")
        d = create_ingestion_dag(
            index_type=index_type,
            array_uri=array_uri,
            source_uri=source_uri,
            source_type=source_type,
            vector_type=vector_type,
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
            mode=mode,
        )
        logger.debug("Submitting ingestion graph")
        d.compute()
        logger.debug("Submitted ingestion graph")
        d.wait()
        consolidate_and_vacuum(array_uri=array_uri, config=config)

        if index_type == "FLAT":
            return FlatIndex(uri=array_uri, dtype=vector_type, config=config)
        elif index_type == "IVF_FLAT":
            return IVFFlatIndex(
                uri=array_uri, dtype=vector_type, memory_budget=1000000, config=config
            )
