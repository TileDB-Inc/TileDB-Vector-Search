import os
import unittest

from array_paths import *
from common import *

import tiledb.vector_search as vs
from tiledb.cloud import groups
from tiledb.cloud.dag import Mode
from tiledb.vector_search.utils import load_fvecs

MINIMUM_ACCURACY = 0.85


class CloudTests(unittest.TestCase):
    flat_index_uri = ""
    ivf_flat_index_uri = ""

    @classmethod
    def setUpClass(cls):
        token = os.getenv("TILEDB_REST_TOKEN")
        if os.getenv("TILEDB_CLOUD_HELPER_VAR"):
            token = os.getenv("TILEDB_CLOUD_HELPER_VAR")
        tiledb.cloud.login(token=token)
        namespace, storage_path, _ = groups._default_ns_path_cred()
        storage_path = storage_path.replace("//", "/").replace("/", "//", 1)
        rand_name = random_name("vector_search")
        test_path = f"tiledb://{namespace}/{storage_path}/{rand_name}"
        cls.flat_index_uri = f"{test_path}/test_cloud_flat_index"
        cls.vamana_index_uri = f"{test_path}/test_cloud_vamana_index"
        cls.ivf_flat_index_uri = f"{test_path}/test_cloud_ivf_flat_index"
        cls.ivf_flat_random_sampling_index_uri = (
            f"{test_path}/test_cloud_ivf_flat_random_sampling_index"
        )

    @classmethod
    def tearDownClass(cls):
        vs.Index.delete_index(uri=cls.flat_index_uri, config=tiledb.cloud.Config())
        vs.Index.delete_index(uri=cls.vamana_index_uri, config=tiledb.cloud.Config())
        vs.Index.delete_index(uri=cls.ivf_flat_index_uri, config=tiledb.cloud.Config())
        vs.Index.delete_index(
            uri=cls.ivf_flat_random_sampling_index_uri, config=tiledb.cloud.Config()
        )

    def run_cloud_test(self, index_uri, index_type, index_class):
        source_uri = "tiledb://TileDB-Inc/sift_10k"
        queries_uri = siftsmall_query_file
        gt_uri = siftsmall_groundtruth_file
        k = 100
        partitions = 100
        nqueries = 100
        nprobe = 20

        queries = load_fvecs(queries_uri)
        gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

        # Test ingest().
        index = vs.ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=source_uri,
            partitions=partitions,
            input_vectors_per_work_item=5000,
            config=tiledb.cloud.Config().dict(),
            mode=Mode.BATCH,
        )
        tiledb_index_uri = groups.info(index_uri).tiledb_uri

        # Test without loading index data into memory.
        index = index_class(
            uri=tiledb_index_uri,
            config=tiledb.cloud.Config().dict(),
            open_for_remote_query_execution=True,
        )
        # Throws if we try to query locally.
        with self.assertRaises(ValueError):
            index.query(queries, k=k, nprobe=nprobe)
        # Succeeeds if we try query with a taskgraph.
        _, result_i = index.query(
            queries=queries,
            k=k,
            nprobe=nprobe,
            driver_mode=Mode.REALTIME,
            num_partitions=2,
        )
        assert accuracy(result_i, gt_i) > MINIMUM_ACCURACY

        # Test query().
        index = index_class(
            uri=tiledb_index_uri,
            config=tiledb.cloud.Config().dict(),
        )
        for driver_mode in [None, Mode.REALTIME]:
            for mode in [None, Mode.LOCAL, Mode.REALTIME]:
                _, result_i = index.query(
                    queries=queries,
                    k=k,
                    nprobe=nprobe,
                    mode=mode,
                    driver_mode=driver_mode,
                    num_partitions=2,
                )
                assert accuracy(result_i, gt_i) > MINIMUM_ACCURACY

        # We now will test for invalid scenarios when setting the query() resources.
        if index_type == "IVF_FLAT":
            resources = {"cpu": "9", "memory": "12Gi", "gpu": 0}
            # Cannot pass resource_class or resources to LOCAL mode or to no mode.
            with self.assertRaises(TypeError):
                index.query(
                    queries, k=k, nprobe=nprobe, mode=Mode.LOCAL, resource_class="large"
                )
            with self.assertRaises(TypeError):
                index.query(
                    queries, k=k, nprobe=nprobe, mode=Mode.LOCAL, resources=resources
                )
            with self.assertRaises(TypeError):
                index.query(queries, k=k, nprobe=nprobe, resource_class="large")
            with self.assertRaises(TypeError):
                index.query(queries, k=k, nprobe=nprobe, resources=resources)
            # Cannot pass resources to REALTIME.
            with self.assertRaises(TypeError):
                index.query(
                    queries, k=k, nprobe=nprobe, mode=Mode.REALTIME, resources=resources
                )
            # Cannot pass both resource_class and resources.
            with self.assertRaises(TypeError):
                index.query(
                    queries,
                    k=k,
                    nprobe=nprobe,
                    mode=Mode.REALTIME,
                    resource_class="large",
                    resources=resources,
                )
            with self.assertRaises(TypeError):
                index.query(
                    queries,
                    k=k,
                    nprobe=nprobe,
                    mode=Mode.BATCH,
                    resource_class="large",
                    resources=resources,
                )

        # Test delete and consolidate_updates.
        index = index_class(
            uri=index_uri,
            config=tiledb.cloud.Config().dict(),
        )
        index.delete(external_id=42)
        _, result_i = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result_i, gt_i) > MINIMUM_ACCURACY

        index = index.consolidate_updates()
        _, result_i = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result_i, gt_i) > MINIMUM_ACCURACY

    def test_cloud_flat(self):
        self.run_cloud_test(CloudTests.flat_index_uri, "FLAT", vs.flat_index.FlatIndex)

    def test_cloud_vamana(self):
        self.run_cloud_test(
            CloudTests.vamana_index_uri, "VAMANA", vs.vamana_index.VamanaIndex
        )

    def test_cloud_ivf_flat(self):
        self.run_cloud_test(
            CloudTests.ivf_flat_index_uri, "IVF_FLAT", vs.ivf_flat_index.IVFFlatIndex
        )

    def test_cloud_ivf_flat_random_sampling(self):
        # NOTE(paris): This was also tested with the following (and also with mode=Mode.BATCH):
        # source_uri = "tiledb://TileDB-Inc/ann_sift1b_raw_vectors_col_major"
        # training_sample_size = 1000000
        source_uri = "tiledb://TileDB-Inc/sift_10k"
        queries_uri = siftsmall_query_file
        gt_uri = siftsmall_groundtruth_file
        index_uri = CloudTests.ivf_flat_random_sampling_index_uri
        k = 100
        nqueries = 100
        nprobe = 20
        max_sampling_tasks = 13
        training_sample_size = 1234

        queries = load_fvecs(queries_uri)
        gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

        index = vs.ingest(
            index_type="IVF_FLAT",
            index_uri=index_uri,
            source_uri=source_uri,
            training_sampling_policy=vs.ingestion.TrainingSamplingPolicy.RANDOM,
            training_sample_size=training_sample_size,
            max_sampling_tasks=max_sampling_tasks,
            config=tiledb.cloud.Config().dict(),
            mode=Mode.BATCH,
        )

        check_training_input_vectors(
            index_uri=index_uri,
            expected_training_sample_size=training_sample_size,
            expected_dimensions=queries.shape[1],
            config=tiledb.cloud.Config().dict(),
        )

        _, result_i = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result_i, gt_i) > MINIMUM_ACCURACY
