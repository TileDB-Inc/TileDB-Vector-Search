from typing import Optional


def ingest(
    array_uri: str,
    source_uri: str,
    source_type: str,
    *,
    config=None,
    namespace: Optional[str] = None,
    size: int = -1,
    partitions: int = -1,
    workers: int = -1,
    verbose: bool = False,
    trace_id: Optional[str] = None,
) -> None:
    """
    Ingest vectors into TileDB.

    :param array_uri: Vector array URI
    :param source_uri: Data source URI
    :param source_type: Type of the source data
    :param config: config dictionary, defaults to None
    :param namespace: TileDB-Cloud namespace, defaults to None
    :param size: Number of input vectors,
        if not provided use the full size of the input dataset
    :param partitions: Number of partitions to load the data with,
        if not provided, is auto-configured based on the dataset size
    :param workers: number of workers for vector ingestion,
        if not provided, is auto-configured based on the dataset size
    :param verbose: verbose logging, defaults to False
    :param trace_id: trace ID for logging, defaults to None
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
        if source_type == "ILEDB_ARRAY":
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
        size: int,
        dimensions: int,
        partitions: int,
        input_vectors_work_items: int,
        vector_type: np.dtype,
        logger: logging.Logger,
    ) -> None:
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

    # --------------------------------------------------------------------
    # UDFs
    # --------------------------------------------------------------------
    def copy_centroids(
        array_uri: str,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ):
        logger = setup(config, verbose)
        group = tiledb.Group(array_uri)
        centroids_uri = group[CENTROIDS_ARRAY_NAME].uri
        src = tiledb.open("s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/centroids.tdb", mode="r")
        dest = tiledb.open(centroids_uri, mode="w")
        dest[:, :] = src[:, :]
        logger.info(src[:, :])

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

                logger.info(
                    f"Reading input vectors start_pos: {part}, end_pos: {part_end}"
                )
                if source_type == "TILEDB_ARRAY":
                    with tiledb.open(source_uri, mode="r") as src_array:
                        in_vectors = np.transpose(
                            src_array[0:dimensions, part:part_end]["a"]
                        ).copy(order="C")
                elif source_type == "U8BIN":
                    vfs = tiledb.VFS()
                    read_size = part_end - part
                    read_offset = part * dimensions + 8
                    with vfs.open(source_uri, "rb") as f:
                        f.seek(read_offset)
                        in_vectors = np.reshape(
                            np.frombuffer(
                                f.read(read_size * dimensions),
                                count=read_size * dimensions,
                                dtype=vector_type,
                            ).astype(np.float32),
                            (read_size, dimensions),
                        )

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
        group = tiledb.Group(array_uri)
        write_array_uri = group[WRITE_ARRAY_NAME].uri
        index_size_array_uri = group[INDEX_SIZE_ARRAY_NAME].uri

        with tiledb.scope_ctx(ctx_or_config=config):
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
        group = tiledb.Group(array_uri)
        index_size_array_uri = group[INDEX_SIZE_ARRAY_NAME].uri
        index_array_uri = group[INDEX_ARRAY_NAME].uri

        with tiledb.scope_ctx(ctx_or_config=config):
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
        group = tiledb.Group(array_uri)
        write_array_uri = group[WRITE_ARRAY_NAME].uri
        index_array_uri = group[INDEX_ARRAY_NAME].uri
        ids_array_uri = group[IDS_ARRAY_NAME].uri
        parts_array_uri = group[PARTS_ARRAY_NAME].uri
        with tiledb.scope_ctx(ctx_or_config=config):
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
        array_uri: str,
        source_uri: str,
        source_type: str,
        vector_type: np.dtype,
        size: int,
        partitions: int,
        dimensions: int,
        input_vectors_per_work_item: int,
        input_vectors_work_items_per_worker: int,
        table_partitions_per_work_item: int,
        table_partitions_work_items_per_worker: int,
        workers: int,
        config: Optional[Mapping[str, Any]] = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
    ) -> dag.DAG:
        d = dag.DAG(
            name="vector-ingestion",
            mode=Mode.BATCH,
            max_workers=workers,
            retry_strategy=models.RetryStrategy(
                limit=1,
                retry_policy="Always",
            ),
        )
        centroids_node = d.submit(copy_centroids,
                                  array_uri=array_uri,
                                  config=config,
                                  verbose=verbose,
                                  trace_id=trace_id,
                                  name="centroids", resources={"cpu": "1", "memory": "4Gi"})
        wait_node = d.submit(print, "ok", name="wait", resources={"cpu": "0.5", "memory": "500Mi"})
        
        task_id = 0
        input_vectors_batch_size = input_vectors_per_work_item * input_vectors_work_items_per_worker
        for i in range(0, size, input_vectors_batch_size):
            start = i
            end = i + input_vectors_batch_size
            if end > size:
                end = size
            ingest_node = d.submit(
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

        compute_indexes_node = d.submit(compute_partition_indexes_udf,
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
            compute_partition_sizes_node = d.submit(compute_partition_sizes_udf,
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
            consolidate_partition_node = d.submit(consolidate_partition_udf,
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

        create_arrays(
            group=group,
            size=size,
            dimensions=dimensions,
            partitions=partitions,
            input_vectors_work_items=input_vectors_work_items,
            vector_type=vector_type,
            logger=logger,
        )
        group.close()

        d = create_ingestion_dag(
            array_uri=array_uri,
            source_uri=source_uri,
            source_type=source_type,
            vector_type=vector_type,
            size=size,
            partitions=partitions,
            dimensions=dimensions,
            input_vectors_per_work_item=input_vectors_per_work_item,
            input_vectors_work_items_per_worker=input_vectors_work_items_per_worker,
            table_partitions_per_work_item=table_partitions_per_work_item,
            table_partitions_work_items_per_worker=table_partitions_work_items_per_worker,
            workers=workers,
            config=config,
            verbose=verbose,
            trace_id=trace_id,
        )
        d.compute()
        d.wait()
