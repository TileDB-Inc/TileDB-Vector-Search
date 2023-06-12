from typing import Optional

from tiledb.cloud.dag import Mode


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
    verbose: bool = False,
    trace_id: Optional[str] = None,
    mode: Mode = Mode.LOCAL,
) -> None:
    """
    Ingest vectors into TileDB.

    :param index_type: Type of vector index (FLAT, IVF_FLAT)
    :param array_uri: Vector array URI
    :param source_uri: Data source URI
    :param source_type: Type of the source data
    :param config: config dictionary, defaults to None
    :param namespace: TileDB-Cloud namespace, defaults to None
    :param size: Number of input vectors,
        if not provided use the full size of the input dataset
    :param partitions: Number of partitions to load the data with,
        if not provided, is auto-configured based on the dataset size
    :param copy_centroids_uri: TileDB array URI to copy centroids from,
        if not provided, centroids are build running kmeans
    :param training_sample_size: vector sample size to train centroids with,
        if not provided, is auto-configured based on the dataset size
    :param workers: number of workers for vector ingestion,
        if not provided, is auto-configured based on the dataset size
    :param verbose: verbose logging, defaults to False
    :param trace_id: trace ID for logging, defaults to None
    :param mode: execution mode, defaults to LOCAL use BATCH for distributed execution
    """
    import enum
    import logging
    import math
    from typing import Any, Mapping, Optional

    import numpy as np
    import tiledb
    from tiledb.cloud import dag
    from tiledb.cloud.dag import Mode
    from tiledb.cloud.rest_api import models
    from tiledb.cloud.utilities import get_logger
    from tiledb.cloud.utilities import set_aws_context

    CENTROIDS_ARRAY_NAME = "centroids.tdb"
    INDEX_ARRAY_NAME = "index.tdb"
    INDEX_SIZE_ARRAY_NAME = "index_size.tdb"
    IDS_ARRAY_NAME = "ids.tdb"
    PARTS_ARRAY_NAME = "parts.tdb"
    WRITE_ARRAY_NAME = "write_temp.tdb"
    VECTORS_PER_WORK_ITEM = 1000000
    MAX_TASKS_PER_STAGE = 100
    CENTRALISED_KMEANS_MAX_SAMPLE_SIZE = 2000000

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

        level = logging.DEBUG if verbose else logging.INFO
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
    ) -> (int, int, np.dtype):
        if source_type == "TILEDB_ARRAY":
            schema = tiledb.ArraySchema.load(source_uri)
            size = schema.domain.dim(1).domain[1] + 1
            dimensions = schema.domain.dim(0).domain[1] + 1
            return size, dimensions, schema.attr("a").dtype
        elif source_type == "U8BIN":
            vfs = tiledb.VFS()
            with vfs.open(source_uri, "rb") as f:
                size = int.from_bytes(f.read(4), "little")
                dimensions = int.from_bytes(f.read(4), "little")
                return size, dimensions, np.uint8
        else:
            raise ValueError(f"Not supported source_type {source_type}")

    def create_arrays(
        group: tiledb.Group,
        index_type: str,
        size: int,
        dimensions: int,
        partitions: int,
        input_vectors_work_items: int,
        vector_type: np.dtype,
        logger: logging.Logger,
    ) -> None:

        if index_type == "FLAT":
            parts_uri = f"{group.uri}/{PARTS_ARRAY_NAME}"
            if not tiledb.array_exists(parts_uri):
                logger.info("Creating parts array")
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
                parts_array_dom = tiledb.Domain(parts_array_rows_dim, parts_array_cols_dim)
                parts_attr = tiledb.Attr(name="values", dtype=vector_type)
                parts_schema = tiledb.ArraySchema(
                    domain=parts_array_dom,
                    sparse=False,
                    attrs=[parts_attr],
                    capacity=int(size / partitions) * dimensions,
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.info(parts_schema)
                tiledb.Array.create(parts_uri, parts_schema)
                group.add(parts_uri, name=PARTS_ARRAY_NAME)

        elif index_type == "FLAT":
            centroids_uri = f"{group.uri}/{CENTROIDS_ARRAY_NAME}"
            index_uri = f"{group.uri}/{INDEX_ARRAY_NAME}"
            index_size_uri = f"{group.uri}/{INDEX_SIZE_ARRAY_NAME}"
            ids_uri = f"{group.uri}/{IDS_ARRAY_NAME}"
            parts_uri = f"{group.uri}/{PARTS_ARRAY_NAME}"
            write_array_uri = f"{group.uri}/{WRITE_ARRAY_NAME}"

            if not tiledb.array_exists(centroids_uri):
                logger.info("Creating centroids array")
                centroids_array_rows_dim = tiledb.Dim(
                    name="rows", domain=(0, dimensions - 1), tile=dimensions, dtype=np.dtype(np.int32)
                )
                centroids_array_cols_dim = tiledb.Dim(
                    name="cols", domain=(0, partitions - 1), tile=partitions, dtype=np.dtype(np.int32)
                )
                centroids_array_dom = tiledb.Domain(
                    centroids_array_rows_dim, centroids_array_cols_dim
                )
                centroids_attr = tiledb.Attr(name="centroids", dtype=np.dtype(np.float32))
                centroids_schema = tiledb.ArraySchema(
                    domain=centroids_array_dom,
                    sparse=False,
                    attrs=[centroids_attr],
                    capacity=dimensions * partitions,
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.info(centroids_schema)
                tiledb.Array.create(centroids_uri, centroids_schema)
                group.add(centroids_uri, name=CENTROIDS_ARRAY_NAME)

            if not tiledb.array_exists(index_uri):
                logger.info("Creating index array")
                index_array_rows_dim = tiledb.Dim(
                    name="rows",
                    domain=(0, partitions - 1),
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
                    cell_order="row-major",
                    tile_order="row-major",
                )
                logger.info(index_schema)
                tiledb.Array.create(index_uri, index_schema)
                group.add(index_uri, name=INDEX_ARRAY_NAME)

            if not tiledb.array_exists(index_size_uri):
                logger.info("Creating index size array")
                index_array_rows_dim = tiledb.Dim(
                    name="rows",
                    domain=(0, partitions - 1),
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
                    cell_order="row-major",
                    tile_order="row-major",
                )
                logger.info(index_schema)
                tiledb.Array.create(index_size_uri, index_schema)
                group.add(index_size_uri, name=INDEX_SIZE_ARRAY_NAME)

            if not tiledb.array_exists(ids_uri):
                logger.info("Creating ids array")
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
                    cell_order="row-major",
                    tile_order="row-major",
                )
                logger.info(ids_schema)
                tiledb.Array.create(ids_uri, ids_schema)
                group.add(ids_uri, name=IDS_ARRAY_NAME)

            if not tiledb.array_exists(parts_uri):
                logger.info("Creating parts array")
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
                parts_array_dom = tiledb.Domain(parts_array_rows_dim, parts_array_cols_dim)
                parts_attr = tiledb.Attr(name="values", dtype=vector_type)
                parts_schema = tiledb.ArraySchema(
                    domain=parts_array_dom,
                    sparse=False,
                    attrs=[parts_attr],
                    capacity=int(size / partitions) * dimensions,
                    cell_order="col-major",
                    tile_order="col-major",
                )
                logger.info(parts_schema)
                tiledb.Array.create(parts_uri, parts_schema)
                group.add(parts_uri, name=PARTS_ARRAY_NAME)

            if not tiledb.array_exists(write_array_uri):
                logger.info("Creating write array")
                kmeans_id_dim = tiledb.Dim(
                    name="kmeans_id",
                    domain=(0, partitions - 1),
                    tile=1,
                    dtype=np.dtype(np.int32),
                )
                object_id_dim = tiledb.Dim(
                    name="object_id", domain=(0, size - 1), tile=size, dtype=np.dtype(np.int32)
                )
                dom = tiledb.Domain(kmeans_id_dim, object_id_dim)
                vector_dtype = [
                    (
                        "",
                        vector_type,
                    )
                    for _ in range(dimensions)
                ]
                vector_attr = tiledb.Attr(name="vector", dtype=np.dtype(vector_dtype))
                schema = tiledb.ArraySchema(
                    domain=dom,
                    sparse=True,
                    attrs=[vector_attr],
                    capacity=max(1000, int(size / partitions / input_vectors_work_items)),
                )
                logger.info(schema)
                tiledb.Array.create(write_array_uri, schema)
                group.add(write_array_uri, name=WRITE_ARRAY_NAME)
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
        trace_id: Optional[str] = None
    ) -> np.array:
        logger = setup(config, verbose)
        logger.info(
            f"Reading input vectors start_pos: {start_pos}, end_pos: {end_pos}"
        )
        if source_type == "TILEDB_ARRAY":
            with tiledb.open(source_uri, mode="r") as src_array:
                return np.transpose(
                    src_array[0:dimensions, start_pos:end_pos]["a"]
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
                    ).astype(np.float32),
                    (read_size, dimensions),
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
        logger.info(f"Copying centroids from: {copy_centroids_uri}, to: {centroids_uri}")
        src = tiledb.open(copy_centroids_uri, mode="r")
        dest = tiledb.open(centroids_uri, mode="w")
        src_centroids = src[:, :]
        dest[:, :] = src_centroids
        logger.info(src_centroids)

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
        init: str = 'random',
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
            sample_vectors = read_input_vectors(source_uri=source_uri,
                                                source_type=source_type,
                                                vector_type=vector_type,
                                                dimensions=dimensions,
                                                start_pos=sample_start_pos,
                                                end_pos=sample_end_pos,
                                                config=config,
                                                verbose=verbose,
                                                trace_id=trace_id)
            verb = 0
            if verbose:
                verb = 3
            logger.info("Start kmeans training")
            km = KMeans(
                n_clusters=partitions, init=init, max_iter=max_iter, verbose=verb, n_init=n_init
            )
            km.fit_predict(sample_vectors)
            logger.info(f"Writing centroids to array {centroids_uri}")
            with tiledb.open(centroids_uri, mode="w") as A:
                A[0:dimensions, 0:partitions] = np.transpose(np.array(km.cluster_centers_))

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
        trace_id: Optional[str] = None
    ) -> np.array:
        logger = setup(config, verbose)
        logger.info("Initialising centroids by reading the first vectors in the source data.")
        with tiledb.scope_ctx(ctx_or_config=config):
            return read_input_vectors(source_uri=source_uri,
                                      source_type=source_type,
                                      vector_type=vector_type,
                                      dimensions=dimensions,
                                      start_pos=0,
                                      end_pos=partitions,
                                      config=config,
                                      verbose=verbose,
                                      trace_id=trace_id)

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

        def generate_new_centroid_per_thread(thread_id, start, end, new_centroid_sums_queue, new_centroid_counts_queue):
            new_centroid_sums = []
            for i in range(len(cents_t)):
                new_centroid_sums.append(cents_t[i])
            new_centroid_count = np.ones(len(cents_t))
            for vector_id in range(start, end):
                if vector_id % 100000 == 0:
                    logger.info(f"Vectors computed: {vector_id}")
                c_id = assignments_t[vector_id]
                if new_centroid_count[c_id] == 1:
                    new_centroid_sums[c_id] = vectors_t[vector_id]
                else:
                    for d in range(dimensions):
                        new_centroid_sums[c_id][d] += vectors_t[vector_id][d]
                new_centroid_count[c_id] += 1
            new_centroid_sums_queue.put(new_centroid_sums)
            new_centroid_counts_queue.put(new_centroid_count)
            logger.info(f"Finished thread: {thread_id}")

        def update_centroids():
            import multiprocessing as mp

            logger.info("Updating centroids based on assignments.")
            logger.info(f"Using {threads} threads.")
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
                worker = mp.Process(target=generate_new_centroid_per_thread, args=(
                    thread_id, i, i + batch_size, new_centroid_sums_queue, new_centroid_counts_queue))
                worker.start()
                workers.append(worker)
                thread_id += 1

            new_centroid_thread_sums_array = []
            new_centroid_thread_counts_array = []
            for i in range(threads):
                new_centroid_thread_sums_array.append(new_centroid_thread_sums[i].get())
                new_centroid_thread_counts_array.append(new_centroid_thread_counts[i].get())
                workers[i].join()

            logger.info(f"Finished all threads, aggregating partial results.")
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
            logger.info("Reading input vectors.")
            vectors = read_input_vectors(source_uri=source_uri,
                                         source_type=source_type,
                                         vector_type=vector_type,
                                         dimensions=dimensions,
                                         start_pos=vector_start_pos,
                                         end_pos=vector_end_pos,
                                         config=config,
                                         verbose=verbose,
                                         trace_id=trace_id)
            logger.info(f"Input centroids: {centroids[0:5]}")
            logger.info("Assigning vectors to centroids")
            km = KMeans()
            km._n_threads = threads
            km.cluster_centers_ = centroids
            assignments = km.predict(vectors)
            logger.info(f"Assignments: {assignments[0:100]}")
            partial_new_centroids = update_centroids()
            logger.info(f"New centroids: {partial_new_centroids[0:5]}")
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
        trace_id: Optional[str] = None):

        import numpy as np
        import tiledb.cloud

        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(array_uri)
            parts_array_uri = group[PARTS_ARRAY_NAME].uri
            target = tiledb.open(parts_array_uri, mode="w")
            logger.info(f"Input vectors start_pos: {start}, end_pos: {end}")

            for part in range(start, end, batch):
                part_end = part + batch
                if part_end > end:
                    part_end = end
                in_vectors = read_input_vectors(source_uri=source_uri,
                                                source_type=source_type,
                                                vector_type=vector_type,
                                                dimensions=dimensions,
                                                start_pos=part,
                                                end_pos=part_end,
                                                config=config,
                                                verbose=verbose,
                                                trace_id=trace_id)

                logger.info(f"Vector read:{len(in_vectors)}")
                logger.info(f"Writing data to array {parts_array_uri}")
                target[0:dimensions, start:end] = np.transpose(in_vectors)
            target.close()

    def write_centroids(
        centroids: np.array,
        array_uri: str,
        partitions: int,
        dimensions: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None):

        with tiledb.scope_ctx(ctx_or_config=config):
            logger = setup(config, verbose)
            group = tiledb.Group(array_uri)
            centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
            logger.info(f"Writing centroids to array {centroids_uri}")
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
        from sklearn.cluster import KMeans
        import numpy as np
        import tiledb.cloud

        logger = setup(config, verbose)
        group = tiledb.Group(array_uri)
        centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
        write_array_uri = group[WRITE_ARRAY_NAME].uri
        vector_dtype = [
            (
                "",
                vector_type,
            )
            for _ in range(dimensions)
        ]
        km = KMeans()
        km._n_threads = threads

        with tiledb.scope_ctx(ctx_or_config=config):
            logger.info(f"Input vectors start_pos: {start}, end_pos: {end}")
            with tiledb.open(centroids_uri, mode="r") as centroids_array:
                logger.info(f"Reading centroids")
                km.cluster_centers_ = np.transpose(
                    centroids_array[0:dimensions, 0:partitions]["centroids"]
                ).copy(order="C")
            target = tiledb.open(write_array_uri, mode="w")

            for part in range(start, end, batch):
                part_end = part + batch
                if part_end > end:
                    part_end = end
                in_vectors = read_input_vectors(source_uri=source_uri,
                                                source_type=source_type,
                                                vector_type=vector_type,
                                                dimensions=dimensions,
                                                start_pos=part,
                                                end_pos=part_end,
                                                config=config,
                                                verbose=verbose,
                                                trace_id=trace_id)

                logger.info(f"Vector read:{len(in_vectors)}")
                logger.info("Assigning vectors to partitions")
                in_vector_partitions = km.predict(in_vectors)
                logger.info("Assignment complete")

                logger.info("Prepare write vectors")
                partitioned_object_ids = []
                for i in range(partitions):
                    partitioned_object_ids.append([])

                object_id = part
                for partition in in_vector_partitions:
                    partitioned_object_ids[partition].append(object_id)
                    object_id += 1

                write_indexes_k = []
                write_indexes_id = []
                write_vectors = []
                for i in range(partitions):
                    for partitioned_object_id in partitioned_object_ids[i]:
                        write_indexes_k.append(i)
                        write_indexes_id.append(partitioned_object_id)
                        write_vectors.append(
                            in_vectors[partitioned_object_id - part]
                            .astype(vector_type)
                            .view(vector_dtype)
                        )
                logger.info(f"Writing data to array {write_array_uri}")
                target[write_indexes_k, write_indexes_id] = {"vector": write_vectors}
            target.close()

    def compute_partition_sizes_udf(
        array_uri: str,
        partition_id_start: int,
        partition_id_end: int,
        batch: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None
    ):
        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(array_uri)
            write_array_uri = group[WRITE_ARRAY_NAME].uri
            index_size_array_uri = group[INDEX_SIZE_ARRAY_NAME].uri
            write_array = tiledb.open(write_array_uri, mode="r")
            index_size_array = tiledb.open(index_size_array_uri, mode="w")
            logger.info(f"Partitions start: {partition_id_start} end: {partition_id_end}")
            for part in range(partition_id_start, partition_id_end, batch):
                part_end = part + batch
                if part_end > partition_id_end:
                    part_end = partition_id_end
                logger.info(f"Read partitions start: {part} end: {part_end}")
                partition = write_array[part:part_end, :]
                sizes = np.zeros(part_end - part).astype(np.uint64)
                for p in partition['kmeans_id']:
                    sizes[p - part] += 1
                logger.info(f"Partition sizes: {sizes}")
                index_size_array[part:part_end] = sizes

    def compute_partition_indexes_udf(
        array_uri: str,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None
    ):
        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(array_uri)
            index_size_array_uri = group[INDEX_SIZE_ARRAY_NAME].uri
            index_array_uri = group[INDEX_ARRAY_NAME].uri
            index_size_array = tiledb.open(index_size_array_uri, mode="r")
            sizes = index_size_array[:]['values']
            sum = 0
            indexes = np.zeros(len(sizes)).astype(np.uint64)
            i = 0
            for size in sizes:
                indexes[i] = sum
                sum += size
                i += 1
            logger.info(f"Partition indexes: {indexes}")
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
        trace_id: Optional[str] = None
    ):
        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            group = tiledb.Group(array_uri)
            logger.info("Consolidating array")
            write_array_uri = group[WRITE_ARRAY_NAME].uri
            index_array_uri = group[INDEX_ARRAY_NAME].uri
            ids_array_uri = group[IDS_ARRAY_NAME].uri
            parts_array_uri = group[PARTS_ARRAY_NAME].uri
            logger.info(f"Partitions start: {partition_id_start} end: {partition_id_end}")
            for part in range(partition_id_start, partition_id_end, batch):
                part_end = part + batch
                if part_end > partition_id_end:
                    part_end = partition_id_end
                logger.info(f"Read partitions start: {part} end: {part_end}")
                with tiledb.open(index_array_uri, 'r') as A:
                    start_pos = A[part]['values']
                logger.info(f"partition start index: {start_pos}")
                logger.info(f"Reading partition data")
                with tiledb.open(write_array_uri, mode="r") as A:
                    partition = A[part:part_end, :]

                partition_size = len(partition['object_id'])
                logger.info(f"Vectors in partition: {partition_size}")
                vectors = []
                for i in range(dimensions):
                    vectors.append([])
                object_ids = []
                pids = partition['object_id']
                tt = 0
                for v in partition['vector']:
                    j = 0
                    for dim in v:
                        vectors[j].append(dim)
                        j += 1
                    object_ids.append(pids[tt])
                    tt += 1

                end_pos = start_pos + partition_size - 1
                logger.info(f"Writing data to array: {parts_array_uri}")
                with tiledb.open(parts_array_uri, mode="w") as A:
                    A[0:dimensions, start_pos:end_pos] = np.array(vectors)
                logger.info(f"Writing data to array: {ids_array_uri}")
                with tiledb.open(ids_array_uri, mode="w") as A:
                    A[start_pos:end_pos] = np.array(object_ids)

    # --------------------------------------------------------------------
    # DAG
    # --------------------------------------------------------------------

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
        else:
            d = dag.DAG(
                name="vector-ingestion",
                mode=Mode.REALTIME,
                max_workers=workers,
                namespace="default",
            )

        submit = d.submit_local
        if mode == Mode.BATCH or mode == Mode.REALTIME:
            submit = d.submit

        input_vectors_batch_size = input_vectors_per_work_item * input_vectors_work_items_per_worker
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
                    batch=input_vectors_per_work_item,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                    name="ingest-" + str(task_id),
                    resources={"cpu": "6", "memory": "32Gi"},
                )
                task_id += 1
            return d
        elif index_type == "IVF_FLAT":
            if copy_centroids_uri is not None:
                centroids_node = submit(copy_centroids,
                                        array_uri=array_uri,
                                        copy_centroids_uri=copy_centroids_uri,
                                        config=config,
                                        verbose=verbose,
                                        trace_id=trace_id,
                                        name="copy-centroids", resources={"cpu": "1", "memory": "2Gi"})
            else:
                if training_sample_size <= CENTRALISED_KMEANS_MAX_SAMPLE_SIZE:
                    centroids_node = submit(centralised_kmeans,
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
                                            name="kmeans", resources={"cpu": "8", "memory": "32Gi"})
                else:
                    internal_centroids_node = submit(init_centroids,
                                                     source_uri=source_uri,
                                                     source_type=source_type,
                                                     vector_type=vector_type,
                                                     partitions=partitions,
                                                     dimensions=dimensions,
                                                     config=config,
                                                     verbose=verbose,
                                                     trace_id=trace_id,
                                                     name="init-centroids",
                                                     resources={"cpu": "1", "memory": "1Gi"})

                    for it in range(5):
                        kmeans_workers = []
                        task_id = 0
                        for i in range(0, training_sample_size, input_vectors_batch_size):
                            start = i
                            end = i + input_vectors_batch_size
                            if end > size:
                                end = size
                            kmeans_workers.append(submit(assign_points_and_partial_new_centroids,
                                                         centroids=internal_centroids_node,
                                                         source_uri=source_uri,
                                                         source_type=source_type,
                                                         vector_type=vector_type,
                                                         partitions=partitions,
                                                         dimensions=dimensions,
                                                         vector_start_pos=start,
                                                         vector_end_pos=end,
                                                         threads=8,
                                                         config=config,
                                                         verbose=verbose,
                                                         trace_id=trace_id,
                                                         name="k-means-part-" + str(task_id),
                                                         resources={"cpu": "8", "memory": "12Gi"}))
                            task_id += 1
                        reducers = []
                        for i in range(0, len(kmeans_workers), 10):
                            reducers.append(submit(compute_new_centroids,
                                                   *kmeans_workers[i:i + 10], name="update-centroids-" + str(i),
                                                   resources={"cpu": "1", "memory": "8Gi"}))
                        internal_centroids_node = submit(compute_new_centroids,
                                                         *reducers, name="update-centroids",
                                                         resources={"cpu": "1", "memory": "8Gi"})
                    centroids_node = submit(write_centroids,
                                            centroids=internal_centroids_node,
                                            array_uri=array_uri,
                                            partitions=partitions,
                                            dimensions=dimensions,
                                            config=config,
                                            verbose=verbose,
                                            trace_id=trace_id,
                                            name="write-centroids",
                                            resources={"cpu": "1", "memory": "2Gi"})

            wait_node = submit(print, "ok", name="wait", resources={"cpu": "0.5", "memory": "500Mi"})

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
                    threads=6,
                    config=config,
                    verbose=verbose,
                    trace_id=trace_id,
                    name="ingest-" + str(task_id),
                    resources={"cpu": "6", "memory": "32Gi"},
                )
                ingest_node.depends_on(centroids_node)
                wait_node.depends_on(ingest_node)
                task_id += 1

            compute_indexes_node = submit(compute_partition_indexes_udf,
                                          array_uri=array_uri,
                                          config=config,
                                          verbose=verbose,
                                          trace_id=trace_id,
                                          name="compute-indexes",
                                          resources={"cpu": "1", "memory": "2Gi"})
            task_id = 0
            partitions_batch = table_partitions_work_items_per_worker * table_partitions_per_work_item
            for i in range(0, partitions, partitions_batch):
                start = i
                end = i + partitions_batch
                if end > partitions:
                    end = partitions
                compute_partition_sizes_node = submit(compute_partition_sizes_udf,
                                                      array_uri=array_uri,
                                                      partition_id_start=start,
                                                      partition_id_end=end,
                                                      batch=table_partitions_per_work_item,
                                                      config=config,
                                                      verbose=verbose,
                                                      trace_id=trace_id,
                                                      name="compute-partition-sizes-" + str(task_id),
                                                      resources={"cpu": "2", "memory": "16Gi"})
                compute_partition_sizes_node.depends_on(wait_node)
                compute_indexes_node.depends_on(compute_partition_sizes_node)
                task_id += 1

            task_id = 0
            for i in range(0, partitions, partitions_batch):
                start = i
                end = i + partitions_batch
                if end > partitions:
                    end = partitions
                consolidate_partition_node = submit(consolidate_partition_udf,
                                                    array_uri=array_uri,
                                                    partition_id_start=start,
                                                    partition_id_end=end,
                                                    batch=table_partitions_per_work_item,
                                                    dimensions=dimensions,
                                                    config=config,
                                                    verbose=verbose,
                                                    trace_id=trace_id,
                                                    name="consolidate-partition-" + str(task_id),
                                                    resources={"cpu": "2", "memory": "24Gi"})
                consolidate_partition_node.depends_on(compute_indexes_node)
                task_id += 1
            return d
        else:
            raise ValueError(f"Not supported index_type {index_type}")

    with tiledb.scope_ctx(ctx_or_config=config):
        logger = setup(config, verbose)
        logger.info("Ingesting Vectors into %r", array_uri)
        try:
            tiledb.group_create(array_uri)
        except tiledb.TileDBError as err:
            message = str(err)
            if "already exists" in message:
                logger.info(f"Group '{array_uri}' already exists")
            else:
                raise err
        group = tiledb.Group(array_uri, "w")

        in_size, dimensions, vector_type = read_source_metadata(
            source_uri=source_uri, source_type=source_type, logger=logger
        )
        if size == -1:
            size = in_size
        if size > in_size:
            size = in_size
        logger.info("Input dataset size %d", size)
        logger.info("Input dataset dimensions %d", dimensions)
        logger.info(f"Vector dimension type {vector_type}")
        if partitions == -1:
            partitions = int(math.sqrt(size))
        if workers == -1:
            workers = 10

        input_vectors_per_work_item = VECTORS_PER_WORK_ITEM
        input_vectors_work_items = int(math.ceil(size / input_vectors_per_work_item))
        input_vectors_work_tasks = input_vectors_work_items
        input_vectors_work_items_per_worker = 1
        if input_vectors_work_tasks > MAX_TASKS_PER_STAGE:
            input_vectors_work_items_per_worker = int(
                math.ceil(input_vectors_work_items / MAX_TASKS_PER_STAGE)
            )
            input_vectors_work_tasks = MAX_TASKS_PER_STAGE
        logger.info("input_vectors_per_work_item %d", input_vectors_per_work_item)
        logger.info("input_vectors_work_items %d", input_vectors_work_items)
        logger.info("input_vectors_work_tasks %d", input_vectors_work_tasks)
        logger.info("input_vectors_work_items_per_worker %d", input_vectors_work_items_per_worker)

        vectors_per_table_partitions = size / partitions
        table_partitions_per_work_item = int(math.ceil(VECTORS_PER_WORK_ITEM / vectors_per_table_partitions))
        table_partitions_work_items = int(math.ceil(partitions / table_partitions_per_work_item))
        table_partitions_work_tasks = table_partitions_work_items
        table_partitions_work_items_per_worker = 1
        if table_partitions_work_tasks > MAX_TASKS_PER_STAGE:
            table_partitions_work_items_per_worker = int(
                math.ceil(table_partitions_work_items / MAX_TASKS_PER_STAGE)
            )
            table_partitions_work_tasks = MAX_TASKS_PER_STAGE
        logger.info("table_partitions_per_work_item %d", table_partitions_per_work_item)
        logger.info("table_partitions_work_items %d", table_partitions_work_items)
        logger.info("table_partitions_work_tasks %d", table_partitions_work_tasks)
        logger.info("table_partitions_work_items_per_worker %d", table_partitions_work_items_per_worker)

        logger.info("Creating arrays")
        create_arrays(
            group=group,
            index_type=index_type,
            size=size,
            dimensions=dimensions,
            partitions=partitions,
            input_vectors_work_items=input_vectors_work_items,
            vector_type=vector_type,
            logger=logger,
        )
        group.close()

        logger.info("Creating ingestion graph")
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
        logger.info("Submitting ingestion graph")
        d.compute()
        logger.info("Submitted ingestion graph")
        d.wait()
