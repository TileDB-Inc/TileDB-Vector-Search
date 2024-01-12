from typing import Any, Mapping, Optional, List
from functools import partial

import numpy as np
from tiledb.cloud.dag import Mode
from tiledb.vector_search.object_api import ObjectIndex
from tiledb.vector_search.object_readers import ObjectReader, ObjectPartition
from tiledb.vector_search.embeddings import ObjectEmbedding


def ingest_embeddings(
    object_index: ObjectIndex,
    embeddings_uri: str,
    external_ids_uri: str,
    index_timestamp: int = None,
    workers: int = -1,
    objects_per_partition = -1,
    object_partitions_per_task: int = -1,
    max_tasks_per_stage: int = -1,
    verbose: bool = False,
    trace_id: Optional[str] = None,
    mode: Mode = Mode.LOCAL,
    config: Optional[Mapping[str, Any]] = None,
    namespace: Optional[str] = None,
    **kwargs,
):
    import logging
    import time
    import multiprocessing
    import math
    import numpy as np
    import tiledb
    from tiledb.cloud import dag
    from tiledb.cloud.rest_api import models
    from tiledb.cloud.utilities import get_logger, set_aws_context

    MAX_TASKS_PER_STAGE = 100
    DEFAULT_IMG_NAME = "3.9-vectorsearch"

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
    # --------------------------------------------------------------------
    # UDFs
    # --------------------------------------------------------------------
    def compute_embeddings_udf(
        object_reader: ObjectReader,
        object_embedding: ObjectEmbedding,
        partitions: List[ObjectPartition],
        embeddings_uri: str,
        external_ids_uri: str,
        index_timestamp: int = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        import numpy as np
        import tiledb.cloud

        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            logger.debug("Loading model...")
            object_embedding.load()
            logger.debug("Model loaded")
            dimensions = object_embedding.dimensions()
            vector_type = object_embedding.vector_type()

            logger.debug("embeddings_uri %s external_ids_uri %s", embeddings_uri, external_ids_uri)
            embeddings_array = tiledb.open(embeddings_uri, "w", timestamp=index_timestamp)
            external_ids_array = tiledb.open(external_ids_uri, "w", timestamp=index_timestamp)
            for partition in partitions:
                logger.debug(f"Computing partition: {partition.index_slice()}")
                logger.debug("Reading objects...")
                objects, metadata = object_reader.read_objects(partition)

                logger.debug("Embedding objects...")
                embeddings = object_embedding.embed(objects, metadata)

                partition_index_slice = partition.index_slice()
                index_start = partition_index_slice[0]
                index_end = partition_index_slice[1]
                logger.debug("Write embeddings index_start: %d, index_end: %d", index_start, index_end)
                embeddings_array[0:dimensions, index_start:index_end] = np.transpose(embeddings).astype(vector_type)
                external_ids_array[index_start:index_end] = objects[object_reader.metadata_array_object_id_dim()]

            embeddings_array.close()
            external_ids_array.close()

    # --------------------------------------------------------------------
    # DAG
    # --------------------------------------------------------------------

    def submit_local(d, func, *args, **kwargs):
        # Drop kwarg
        kwargs.pop("image_name", None)
        kwargs.pop("resources", None)
        return d.submit_local(func, *args, **kwargs)

    def create_dag(
        object_index: ObjectIndex,
        embeddings_uri: str,
        external_ids_uri: str,
        partitions: List[ObjectPartition],
        object_partitions_per_worker: int,
        object_work_tasks: int,
        index_timestamp: int = None,
        workers: int = -1,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        mode: Mode = Mode.LOCAL,
        config: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
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

        task_id = 0
        num_partitions = len(partitions)
        for i in range(0, num_partitions, object_partitions_per_worker):
            start = i
            end = i + object_partitions_per_worker
            if end > num_partitions:
                end = num_partitions
            submit(
                compute_embeddings_udf,
                object_reader=object_index.object_reader,
                object_embedding=object_index.embedding,
                partitions=partitions[start:end],
                embeddings_uri=embeddings_uri,
                external_ids_uri=external_ids_uri,
                index_timestamp=index_timestamp,
                verbose=verbose,
                trace_id=trace_id,
                config=config,
                name="generate_embeddings-" + str(task_id),
                resources={"cpu": str(threads), "memory": "16Gi"},
                image_name=DEFAULT_IMG_NAME,
            )
            task_id += 1
        return d

    # --------------------------------------------------------------------
    # End internal function definitions
    # --------------------------------------------------------------------

    with tiledb.scope_ctx(ctx_or_config=config):
        logger = setup(config, verbose)
        logger.debug("Generating embeddings")
        if index_timestamp is None:
            index_timestamp = int(time.time() * 1000)

        print(f"objects_per_partition: {objects_per_partition}")
        partitions = object_index.object_reader.get_partitions(partition_size=objects_per_partition)
        object_partitions = len(partitions)
        object_partitions_per_worker = 1
        if max_tasks_per_stage == -1:
            max_tasks_per_stage = MAX_TASKS_PER_STAGE
        object_work_tasks = object_partitions
        if object_partitions > max_tasks_per_stage:
            object_partitions_per_worker = int(
                math.ceil(object_partitions / max_tasks_per_stage)
            )
            object_work_tasks = max_tasks_per_stage
        logger.debug("object_partitions %d", object_partitions)
        logger.debug("object_work_tasks %d", object_work_tasks)
        logger.debug("object_partitions_per_worker %d", object_partitions_per_worker)
        if mode == Mode.BATCH:
            if workers == -1:
                workers = 10
        else:
            if workers == -1:
                workers = 1

        logger.debug("Creating ingestion graph")
        d = create_dag(
            object_index=object_index,
            embeddings_uri=embeddings_uri,
            external_ids_uri=external_ids_uri,
            partitions=partitions,
            object_partitions_per_worker=object_partitions_per_worker,
            object_work_tasks=object_work_tasks,
            index_timestamp=index_timestamp,
            workers=workers,
            verbose=verbose,
            trace_id=trace_id,
            mode=mode,
            config=config,
            namespace=namespace,
        )
        logger.debug("Submitting ingestion graph")
        d.compute()
        logger.debug("Submitted ingestion graph")
        d.wait()
