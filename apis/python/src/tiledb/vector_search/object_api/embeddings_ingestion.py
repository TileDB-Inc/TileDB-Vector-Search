from typing import Any, Mapping, Optional, List, Dict
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
    metadata_array_uri: str = None,
    index_timestamp: int = None,
    workers: int = -1,
    worker_resources: Dict = None,
    worker_image: str = None,
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
    DEFAULT_WORKER_RESOURCES = { "cpu": "1", "memory": "4Gi"}

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
        object_reader_source_code: str,
        object_reader_class_name: str,
        object_reader_kwargs: Dict,
        object_embedding_source_code: str,
        object_embedding_class_name: str,
        object_embedding_kwargs: Dict,
        partition_dicts: List[Dict],
        embeddings_uri: str,
        external_ids_uri: str,
        metadata_array_uri: str = None,
        index_timestamp: int = None,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        import numpy as np
        import tiledb

        def instantiate_object(code, class_name, **kwargs):
            import random
            import string
            import os
            import sys
            import importlib.util

            temp_file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
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

        logger = setup(config, verbose)
        object_reader = instantiate_object(
            code=object_reader_source_code,
            class_name=object_reader_class_name,
            **object_reader_kwargs
        )
        object_embedding = instantiate_object(
            code=object_embedding_source_code,
            class_name=object_embedding_class_name,
            **object_embedding_kwargs
        )
        with tiledb.scope_ctx(ctx_or_config=config):
            logger.debug("Loading model...")
            object_embedding.load()
            logger.debug("Model loaded")
            dimensions = object_embedding.dimensions()
            vector_type = object_embedding.vector_type()

            logger.debug("embeddings_uri %s external_ids_uri %s", embeddings_uri, external_ids_uri)
            embeddings_array = tiledb.open(embeddings_uri, "w", timestamp=index_timestamp)
            external_ids_array = tiledb.open(external_ids_uri, "w", timestamp=index_timestamp)
            if metadata_array_uri is not None:
                metadata_array = tiledb.open(metadata_array_uri, "w", timestamp=index_timestamp)
            for partition_dict in partition_dicts:
                partition = instantiate_object(
                    code=object_reader_source_code,
                    class_name=object_reader.partition_class_name(),
                    **partition_dict
                )
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
                external_ids_array[index_start:index_end] = objects["external_id"]
                if metadata_array_uri is not None:
                    external_ids = metadata.pop("external_id", None)
                    metadata_array[external_ids] = metadata

            embeddings_array.close()
            external_ids_array.close()
            if metadata_array_uri is not None:
                metadata_array.close()

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
        metadata_array_uri: str = None,
        index_timestamp: int = None,
        workers: int = -1,
        worker_resources: Dict = None,
        worker_image: str = DEFAULT_IMG_NAME,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        mode: Mode = Mode.LOCAL,
        config: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> dag.DAG:
        if mode == Mode.BATCH:
            d = dag.DAG(
                name="embedding-generation",
                mode=Mode.BATCH,
                max_workers=workers,
                retry_strategy=models.RetryStrategy(
                    limit=1,
                    retry_policy="Always",
                ),
            )
        else:
            d = dag.DAG(
                name="embedding-generation",
                mode=Mode.REALTIME,
                max_workers=workers,
                namespace="default",
            )

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
            partition_dicts = []
            for partition in partitions[start:end]:
                partition_dicts.append(partition.init_kwargs())
            submit(
                compute_embeddings_udf,
                object_reader_source_code=object_index.object_reader_source_code,
                object_reader_class_name=object_index.object_reader_class_name,
                object_reader_kwargs=object_index.object_reader_kwargs,
                object_embedding_source_code=object_index.embedding_source_code,
                object_embedding_class_name=object_index.embedding_class_name,
                object_embedding_kwargs=object_index.embedding_kwargs,
                partition_dicts=partition_dicts,
                embeddings_uri=embeddings_uri,
                external_ids_uri=external_ids_uri,
                metadata_array_uri=metadata_array_uri,
                index_timestamp=index_timestamp,
                verbose=verbose,
                trace_id=trace_id,
                config=config,
                name="generate_embeddings-" + str(task_id),
                resources=worker_resources,
                image_name=worker_image,
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
        logger.debug("objects_per_partition %d", objects_per_partition)
        logger.debug("object_partitions %d", object_partitions)
        logger.debug("object_work_tasks %d", object_work_tasks)
        logger.debug("object_partitions_per_worker %d", object_partitions_per_worker)
        if mode == Mode.BATCH:
            if workers == -1:
                workers = 10
            if worker_resources is None:
                worker_resources=DEFAULT_WORKER_RESOURCES
        else:
            if workers == -1:
                workers = 1
        if worker_image is None:
            worker_image=DEFAULT_IMG_NAME

        logger.debug("Creating ingestion graph")
        d = create_dag(
            object_index=object_index,
            embeddings_uri=embeddings_uri,
            external_ids_uri=external_ids_uri,
            partitions=partitions,
            object_partitions_per_worker=object_partitions_per_worker,
            object_work_tasks=object_work_tasks,
            metadata_array_uri=metadata_array_uri,
            index_timestamp=index_timestamp,
            workers=workers,
            worker_resources=worker_resources,
            worker_image=worker_image,
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
