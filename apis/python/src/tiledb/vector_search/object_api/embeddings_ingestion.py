from functools import partial
from typing import Any, Dict, List, Mapping, Optional

from tiledb.cloud.dag import Mode


def ingest_embeddings_with_driver(
    object_index_uri: str,
    embeddings_uri: str,
    metadata_array_uri: str = None,
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
    **kwargs,
):
    def ingest_embeddings(
        object_index_uri: str,
        embeddings_uri: str,
        metadata_array_uri: str = None,
        index_timestamp: int = None,
        workers: int = -1,
        worker_resources: Dict = None,
        worker_image: str = None,
        extra_worker_modules: Optional[List[str]] = None,
        extra_driver_modules: Optional[List[str]] = None,
        worker_access_credentials_name: str = None,
        max_tasks_per_stage: int = -1,
        verbose: bool = False,
        trace_id: Optional[str] = None,
        embeddings_generation_mode: Mode = Mode.LOCAL,
        vector_indexing_mode: Mode = Mode.LOCAL,
        config: Optional[Mapping[str, Any]] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ):
        import tiledb

        def install_extra_worker_modules():
            if extra_driver_modules is not None:
                import os
                import subprocess
                import sys

                sys.path.insert(0, "/home/udf/.local/bin")
                sys.path.insert(0, "/home/udf/.local/lib/python3.9/site-packages")
                os.environ["PATH"] = f"/home/udf/.local/bin:{os.environ['PATH']}"
                for module in extra_driver_modules:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", module]
                    )

        install_extra_worker_modules()
        import importlib

        importlib.reload(tiledb)

        import logging
        import math
        import time

        import tiledb
        from tiledb.cloud import dag
        from tiledb.cloud.rest_api import models
        from tiledb.cloud.utilities import get_logger
        from tiledb.vector_search import ingest
        from tiledb.vector_search.object_api import ObjectIndex
        from tiledb.vector_search.object_readers import ObjectPartition

        MAX_TASKS_PER_STAGE = 100
        DEFAULT_IMG_NAME = "3.9-vectorsearch"
        DEFAULT_WORKER_RESOURCES = {"cpu": "1", "memory": "4Gi"}

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
            metadata_array_uri: str = None,
            index_timestamp: int = None,
            verbose: bool = False,
            trace_id: Optional[str] = None,
            config: Optional[Mapping[str, Any]] = None,
            extra_worker_modules: Optional[List[str]] = None,
        ):
            def install_extra_driver_modules():
                if extra_worker_modules is not None:
                    import os
                    import subprocess
                    import sys

                    sys.path.insert(0, "/home/udf/.local/bin")
                    sys.path.insert(0, "/home/udf/.local/lib/python3.9/site-packages")
                    os.environ["PATH"] = f"/home/udf/.local/bin:{os.environ['PATH']}"
                    for module in extra_worker_modules:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", module]
                        )

            install_extra_driver_modules()

            import numpy as np

            import tiledb

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

            logger = setup(config, verbose)
            object_reader = instantiate_object(
                code=object_reader_source_code,
                class_name=object_reader_class_name,
                **object_reader_kwargs,
            )
            object_embedding = instantiate_object(
                code=object_embedding_source_code,
                class_name=object_embedding_class_name,
                **object_embedding_kwargs,
            )
            with tiledb.scope_ctx(ctx_or_config=config):
                logger.debug("Loading model...")
                object_embedding.load()
                logger.debug("Model loaded")
                object_embedding.dimensions()
                vector_type = object_embedding.vector_type()

                logger.debug("embeddings_uri %s", embeddings_uri)
                embeddings_array = tiledb.open(
                    embeddings_uri, "w", timestamp=index_timestamp
                )
                if metadata_array_uri is not None:
                    metadata_array = tiledb.open(
                        metadata_array_uri, "w", timestamp=index_timestamp
                    )
                for partition_dict in partition_dicts:
                    partition = instantiate_object(
                        code=object_reader_source_code,
                        class_name=object_reader.partition_class_name(),
                        **partition_dict,
                    )
                    partition_id = partition.id()
                    logger.debug(f"Computing partition: {partition_id}")
                    logger.debug("Reading objects...")
                    objects, metadata = object_reader.read_objects(partition)

                    logger.debug("Embedding objects...")
                    embeddings = object_embedding.embed(objects, metadata)

                    logger.debug("Write embeddings partition_id: %d", partition_id)
                    embeddings_flattened = np.empty(1, dtype="O")
                    embeddings_flattened[0] = embeddings.astype(vector_type).flatten()
                    embeddings_shape = np.empty(1, dtype="O")
                    embeddings_shape[0] = np.array(embeddings.shape, dtype=np.uint32)
                    external_ids = np.empty(1, dtype="O")
                    external_ids[0] = objects["external_id"].astype(np.uint64)
                    embeddings_array[partition_id] = {
                        "vectors": embeddings_flattened,
                        "vectors_shape": embeddings_shape,
                        "external_ids": external_ids,
                    }
                    if metadata_array_uri is not None:
                        external_ids = metadata.pop("external_id", None)
                        metadata_array[external_ids] = metadata

                embeddings_array.close()
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
            ob_index: ObjectIndex,
            embeddings_uri: str,
            partitions: List[ObjectPartition],
            object_partitions_per_worker: int,
            object_work_tasks: int,
            metadata_array_uri: str = None,
            index_timestamp: int = None,
            workers: int = -1,
            worker_resources: Dict = None,
            worker_image: str = DEFAULT_IMG_NAME,
            extra_worker_modules: Optional[List[str]] = None,
            worker_access_credentials_name: str = None,
            verbose: bool = False,
            trace_id: Optional[str] = None,
            embeddings_generation_mode: Mode = Mode.LOCAL,
            config: Optional[Mapping[str, Any]] = None,
            namespace: Optional[str] = None,
        ) -> dag.DAG:
            if embeddings_generation_mode == Mode.BATCH:
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
            if (
                embeddings_generation_mode == Mode.BATCH
                or embeddings_generation_mode == Mode.REALTIME
            ):
                submit = d.submit
            extra_udf_worker_modules = None
            if (
                embeddings_generation_mode == Mode.BATCH
                or embeddings_generation_mode == Mode.REALTIME
            ):
                submit = d.submit
                extra_udf_worker_modules = extra_worker_modules

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
                compute_embeddings_udf_kwargs = {}
                if embeddings_generation_mode == Mode.BATCH:
                    compute_embeddings_udf_kwargs[
                        "access_credentials_name"
                    ] = worker_access_credentials_name
                submit(
                    compute_embeddings_udf,
                    object_reader_source_code=ob_index.object_reader_source_code,
                    object_reader_class_name=ob_index.object_reader_class_name,
                    object_reader_kwargs=ob_index.object_reader_kwargs,
                    object_embedding_source_code=ob_index.embedding_source_code,
                    object_embedding_class_name=ob_index.embedding_class_name,
                    object_embedding_kwargs=ob_index.embedding_kwargs,
                    partition_dicts=partition_dicts,
                    embeddings_uri=embeddings_uri,
                    metadata_array_uri=metadata_array_uri,
                    index_timestamp=index_timestamp,
                    verbose=verbose,
                    trace_id=trace_id,
                    config=config,
                    extra_worker_modules=extra_udf_worker_modules,
                    name="generate-embeddings-" + str(task_id),
                    resources=worker_resources,
                    image_name=worker_image,
                    **compute_embeddings_udf_kwargs,
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

            from tiledb.vector_search.object_api import object_index

            ob_index = object_index.ObjectIndex(object_index_uri, config=config)
            partitions = ob_index.object_reader.get_partitions(**kwargs)
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
            logger.debug(
                "object_partitions_per_worker %d", object_partitions_per_worker
            )
            if embeddings_generation_mode == Mode.BATCH:
                if workers == -1:
                    workers = 10
                if worker_resources is None:
                    worker_resources = DEFAULT_WORKER_RESOURCES
            else:
                if workers == -1:
                    workers = 1
            if worker_image is None:
                worker_image = DEFAULT_IMG_NAME

            logger.debug("Creating ingestion graph")
            d = create_dag(
                ob_index=ob_index,
                embeddings_uri=embeddings_uri,
                partitions=partitions,
                object_partitions_per_worker=object_partitions_per_worker,
                object_work_tasks=object_work_tasks,
                metadata_array_uri=metadata_array_uri,
                index_timestamp=index_timestamp,
                workers=workers,
                worker_resources=worker_resources,
                worker_image=worker_image,
                extra_worker_modules=extra_worker_modules,
                worker_access_credentials_name=worker_access_credentials_name,
                verbose=verbose,
                trace_id=trace_id,
                embeddings_generation_mode=embeddings_generation_mode,
                config=config,
                namespace=namespace,
            )
            logger.debug("Submitting ingestion graph")
            d.compute()
            logger.debug("Submitted ingestion graph")
            d.wait()

            ob_index.index = ingest(
                index_type=ob_index.index_type,
                index_uri=ob_index.uri,
                source_uri=embeddings_uri,
                source_type="TILEDB_PARTITIONED_ARRAY",
                external_ids_uri=embeddings_uri,
                external_ids_type="TILEDB_PARTITIONED_ARRAY",
                index_timestamp=index_timestamp,
                storage_version=ob_index.index.storage_version,
                config=config,
                mode=vector_indexing_mode,
                **kwargs,
            )

    def submit_local(d, func, *args, **kwargs):
        # Drop kwarg
        kwargs.pop("image_name", None)
        kwargs.pop("resources", None)
        return d.submit_local(func, *args, **kwargs)

    from tiledb.cloud import dag

    if embeddings_generation_driver_mode == Mode.BATCH:
        d = dag.DAG(
            mode=Mode.BATCH,
            name="ingest-embeddings-driver",
            max_workers=1,
        )
    else:
        d = dag.DAG(
            mode=Mode.REALTIME,
            name="ingest-embeddings-driver",
            max_workers=1,
            namespace="default",
        )
    submit = partial(submit_local, d)
    if (
        embeddings_generation_driver_mode == Mode.BATCH
        or embeddings_generation_driver_mode == Mode.REALTIME
    ):
        submit = d.submit
    driver_access_credentials_name_kwargs = {}
    if embeddings_generation_mode == Mode.BATCH:
        driver_access_credentials_name_kwargs[
            "access_credentials_name"
        ] = worker_access_credentials_name
    submit(
        ingest_embeddings,
        object_index_uri=object_index_uri,
        embeddings_uri=embeddings_uri,
        metadata_array_uri=metadata_array_uri,
        index_timestamp=index_timestamp,
        max_tasks_per_stage=max_tasks_per_stage,
        workers=workers,
        worker_resources=worker_resources,
        worker_image=worker_image,
        extra_worker_modules=extra_worker_modules,
        extra_driver_modules=extra_driver_modules,
        worker_access_credentials_name=worker_access_credentials_name,
        verbose=verbose,
        trace_id=trace_id,
        embeddings_generation_mode=embeddings_generation_mode,
        vector_indexing_mode=vector_indexing_mode,
        config=config,
        **kwargs,
        name="ingest-embeddings-driver",
        resources={"cpu": "1", "memory": "4Gi"}
        if driver_resources is None
        else driver_resources,
        image_name="vectorsearch" if driver_image is None else driver_image,
        **driver_access_credentials_name_kwargs,
    )
    d.compute()
    d.wait()
