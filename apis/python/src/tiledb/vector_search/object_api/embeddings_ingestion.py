from functools import partial
from typing import Any, Dict, List, Mapping, Optional

from tiledb.cloud.dag import Mode


def ingest_embeddings_with_driver(
    object_index_uri: str,
    use_updates_array: bool,
    embeddings_array_uri: str = None,
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
    environment_variables: Dict = {},
    **kwargs,
):
    def ingest_embeddings(
        object_index_uri: str,
        use_updates_array: bool,
        embeddings_array_uri: str = None,
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
        environment_variables: Dict = {},
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
            object_index_uri: str,
            partition_dicts: List[Dict],
            use_updates_array: bool,
            embeddings_array_uri: str = None,
            metadata_array_uri: str = None,
            index_timestamp: int = None,
            verbose: bool = False,
            trace_id: Optional[str] = None,
            config: Optional[Mapping[str, Any]] = None,
            extra_worker_modules: Optional[List[str]] = None,
            environment_variables: Dict = {},
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

            import os

            import numpy as np

            import tiledb
            from tiledb.vector_search.object_api import ObjectIndex

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
            obj_index = ObjectIndex(
                object_index_uri,
                config=config,
                environment_variables=environment_variables,
                load_embedding=False,
                load_metadata_in_memory=False,
            )
            object_reader = obj_index.object_reader
            object_embedding = obj_index.embedding
            for var, val in environment_variables.items():
                os.environ[var] = val
            with tiledb.scope_ctx(ctx_or_config=config):
                logger.debug("Loading model...")
                object_embedding.load()
                logger.debug("Model loaded")
                object_embedding.dimensions()
                vector_type = object_embedding.vector_type()

                if not use_updates_array:
                    logger.debug("embeddings_uri %s", embeddings_array_uri)
                    embeddings_array = tiledb.open(
                        embeddings_array_uri, "w", timestamp=index_timestamp
                    )

                if metadata_array_uri is not None:
                    metadata_array = tiledb.open(
                        metadata_array_uri, "w", timestamp=index_timestamp
                    )
                for partition_dict in partition_dicts:
                    partition = instantiate_object(
                        code=obj_index.object_reader_source_code,
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
                    if use_updates_array:
                        vectors = np.empty(embeddings.shape[0], dtype="O")
                        for i in range(embeddings.shape[0]):
                            vectors[i] = embeddings[i].astype(vector_type)
                        obj_index.index.update_batch(
                            vectors=vectors,
                            external_ids=objects["external_id"].astype(np.uint64),
                        )
                    else:
                        embeddings_flattened = np.empty(1, dtype="O")
                        embeddings_flattened[0] = embeddings.astype(
                            vector_type
                        ).flatten()
                        embeddings_shape = np.empty(1, dtype="O")
                        embeddings_shape[0] = np.array(
                            embeddings.shape, dtype=np.uint32
                        )
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

                if not use_updates_array:
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
            obj_index: ObjectIndex,
            use_updates_array: bool,
            partitions: List[ObjectPartition],
            object_partitions_per_worker: int,
            object_work_tasks: int,
            embeddings_array_uri: str = None,
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
            environment_variables: Dict = {},
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
                    namespace=namespace,
                )
            else:
                d = dag.DAG(
                    name="embedding-generation",
                    mode=Mode.REALTIME,
                    max_workers=workers,
                    # TODO: `default` is not an actual namespace. This is a temp fix to
                    # be able to run DAGs locally.
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
                    object_index_uri=obj_index.uri,
                    partition_dicts=partition_dicts,
                    use_updates_array=use_updates_array,
                    embeddings_array_uri=embeddings_array_uri,
                    metadata_array_uri=metadata_array_uri,
                    index_timestamp=index_timestamp,
                    verbose=verbose,
                    trace_id=trace_id,
                    config=config,
                    environment_variables=environment_variables,
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

        import os

        for var, val in environment_variables.items():
            os.environ[var] = val
        with tiledb.scope_ctx(ctx_or_config=config):
            logger = setup(config, verbose)
            logger.debug("Generating embeddings")
            if index_timestamp is None:
                index_timestamp = int(time.time() * 1000)

            from tiledb.vector_search.object_api import object_index

            obj_index = object_index.ObjectIndex(
                object_index_uri,
                config=config,
                environment_variables=environment_variables,
                load_embedding=False,
                load_metadata_in_memory=False,
            )
            partitions = obj_index.object_reader.get_partitions(**kwargs)
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
                obj_index=obj_index,
                use_updates_array=use_updates_array,
                partitions=partitions,
                object_partitions_per_worker=object_partitions_per_worker,
                object_work_tasks=object_work_tasks,
                embeddings_array_uri=embeddings_array_uri,
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
                environment_variables=environment_variables,
                namespace=namespace,
            )
            logger.debug("Submitting ingestion graph")
            d.compute()
            logger.debug("Submitted ingestion graph")
            d.wait()

            if use_updates_array:
                obj_index.index.consolidate_updates(
                    mode=vector_indexing_mode,
                    **kwargs,
                )
            else:
                obj_index.index = ingest(
                    index_type=obj_index.index_type,
                    index_uri=obj_index.uri,
                    source_uri=embeddings_array_uri,
                    source_type="TILEDB_PARTITIONED_ARRAY",
                    external_ids_uri=embeddings_array_uri,
                    external_ids_type="TILEDB_PARTITIONED_ARRAY",
                    index_timestamp=index_timestamp,
                    storage_version=obj_index.index.storage_version,
                    config=config,
                    namespace=namespace,
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
            namespace=namespace,
        )
    else:
        d = dag.DAG(
            mode=Mode.REALTIME,
            name="ingest-embeddings-driver",
            max_workers=1,
            # TODO: `default` is not an actual namespace. This is a temp fix to
            # be able to run DAGs locally.
            namespace="default",
        )
    submit = partial(submit_local, d)
    if (
        embeddings_generation_driver_mode == Mode.BATCH
        or embeddings_generation_driver_mode == Mode.REALTIME
    ):
        submit = d.submit
    driver_access_credentials_name_kwargs = {}
    if embeddings_generation_driver_mode == Mode.BATCH:
        driver_access_credentials_name_kwargs[
            "access_credentials_name"
        ] = worker_access_credentials_name
    submit(
        ingest_embeddings,
        object_index_uri=object_index_uri,
        use_updates_array=use_updates_array,
        embeddings_array_uri=embeddings_array_uri,
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
        environment_variables=environment_variables,
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
