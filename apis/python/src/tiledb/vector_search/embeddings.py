from typing import Any, Mapping, Optional, Callable
from functools import partial

import numpy as np
from tiledb.cloud.dag import Mode


def generate_embeddings(
    object_array_uri: str,
    embeddings_uri: str,
    external_ids_uri: str,
    dimensions: int,
    vector_type: np.dtype,
    object_id_dim: int,
    load_embedding_model_udf: str,
    embedding_udf: str,
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
    import logging
    import time
    import multiprocessing
    import math
    import numpy as np
    import tiledb
    from tiledb.cloud import dag
    from tiledb.cloud.rest_api import models
    from tiledb.cloud.utilities import get_logger, set_aws_context

    OBJECTS_PER_WORK_ITEM = 100
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
        object_array_uri: str,
        embeddings_uri: str,
        external_ids_uri: str,
        dimensions: int,
        vector_type: np.dtype,
        object_id_dim: str,
        load_embedding_model_udf: str,
        embedding_udf: str,
        start: int,
        end: int,
        batch: int,
        object_array_timestamp=None,
        index_timestamp: int = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        import base64
        import cloudpickle
        import numpy as np
        import tiledb.cloud

        logger = setup(config, verbose)
        with tiledb.scope_ctx(ctx_or_config=config):
            load_embedding_model = cloudpickle.loads(base64.b64decode(load_embedding_model_udf))
            embedding = cloudpickle.loads(base64.b64decode(embedding_udf))

            logger.debug("Loading model...")
            model = load_embedding_model()
            logger.debug("Model loaded")

            logger.debug("embeddings_uri %s external_ids_uri %s", embeddings_uri, external_ids_uri)
            embeddings_array = tiledb.open(embeddings_uri, "w", timestamp=index_timestamp)
            external_ids_array = tiledb.open(external_ids_uri, "w", timestamp=index_timestamp)
            object_array = tiledb.open(object_array_uri, "r", timestamp=object_array_timestamp)
            for part in range(start, end, batch):
                part_end = part + batch
                if part_end > end:
                    part_end = end

                logger.debug("Loading objects start_pos: %d, end_pos: %d", part, part_end)
                data = object_array[part:part_end]

                logger.debug("Embedding objects start_pos: %d, end_pos: %d", part, part_end)
                embeddings = embedding(model, data)

                logger.debug("Write embeddings start_pos: %d, end_pos: %d", part, part_end)
                embeddings_array[0:dimensions, part:part_end] = np.transpose(embeddings).astype(vector_type)
                external_ids_array[part:part_end] = data[object_id_dim]
            object_array.close()
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
        object_array_uri: str,
        embeddings_uri: str,
        external_ids_uri: str,
        dimensions: int,
        vector_type: np.dtype,
        object_id_dim: int,
        load_embedding_model_udf: str,
        embedding_udf: str,
        objects_per_work_item: int,
        object_work_items_per_worker: int,
        size: int,
        object_array_timestamp=None,
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

        object_batch_size = (
            objects_per_work_item * object_work_items_per_worker
        )

        task_id = 0
        for i in range(0, size, object_batch_size):
            start = i
            end = i + object_batch_size
            if end > size:
                end = size
            submit(
                compute_embeddings_udf,
                object_array_uri=object_array_uri,
                embeddings_uri=embeddings_uri,
                external_ids_uri=external_ids_uri,
                dimensions=dimensions,
                vector_type=vector_type,
                object_id_dim=object_id_dim,
                load_embedding_model_udf=load_embedding_model_udf,
                embedding_udf=embedding_udf,
                start=start,
                end=end,
                batch=objects_per_work_item,
                object_array_timestamp=object_array_timestamp,
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
        logger.debug("Generating embeddings for %r", object_array_uri)
        if index_timestamp is None:
            index_timestamp = int(time.time() * 1000)

        with tiledb.open(object_array_uri, mode='r', timestamp=object_array_timestamp) as object_array:
            nonempty_object_array_domain = object_array.nonempty_domain()[0]
            size = nonempty_object_array_domain[1] + 1 - nonempty_object_array_domain[0]

        if objects_per_work_item == -1:
            objects_per_work_item = OBJECTS_PER_WORK_ITEM
        object_work_items = int(math.ceil(size / objects_per_work_item))
        object_work_tasks = object_work_items
        object_work_items_per_worker = 1
        if max_tasks_per_stage == -1:
            max_tasks_per_stage = MAX_TASKS_PER_STAGE
        if object_work_tasks > max_tasks_per_stage:
            object_work_items_per_worker = int(
                math.ceil(object_work_items / max_tasks_per_stage)
            )
            object_work_tasks = max_tasks_per_stage
        logger.debug("objects_per_work_item %d", objects_per_work_item)
        logger.debug("object_work_items %d", object_work_items)
        logger.debug("object_work_tasks %d", object_work_tasks)
        logger.debug(
            "object_work_items_per_worker %d",
            object_work_items_per_worker,
        )
        if mode == Mode.BATCH:
            if workers == -1:
                workers = 10
        else:
            workers = 1

        logger.debug("Creating ingestion graph")
        d = create_dag(
            object_array_uri=object_array_uri,
            embeddings_uri=embeddings_uri,
            external_ids_uri=external_ids_uri,
            dimensions=dimensions,
            vector_type=vector_type,
            object_id_dim=object_id_dim,
            load_embedding_model_udf=load_embedding_model_udf,
            embedding_udf=embedding_udf,
            objects_per_work_item=objects_per_work_item,
            object_work_items_per_worker=object_work_items_per_worker,
            size=size,
            object_array_timestamp=object_array_timestamp,
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
