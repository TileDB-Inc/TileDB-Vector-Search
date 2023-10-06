import unittest
import tiledb.vector_search as vs
import os

from common import *
from tiledb.cloud import groups
from tiledb.cloud.dag import Mode
from tiledb.vector_search.utils import load_fvecs

MINIMUM_ACCURACY = 0.85

class CloudTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        tiledb.cloud.login(token=os.getenv("TILEDB_REST_TOKEN"))
        self.namespace, storage_path, _ = groups._default_ns_path_cred()
        storage_path = storage_path.replace("//", "/").replace("/", "//", 1)
        rand_name = random_name("vector_search")
        self.test_path = f"tiledb://{self.namespace}/{storage_path}/{rand_name}"
        self.config = tiledb.cloud.Config()
        self.config_dict = self.config.dict()
        self.flat_index_uri = f"{self.test_path}/test_flat_array"
        self.ivf_flat_index_uri = f"{self.test_path}/test_ivf_flat_array"

    def tearDown(self):
        vs.Index.delete_index(uri=self.flat_index_uri, config=self.config)
        vs.Index.delete_index(uri=self.ivf_flat_index_uri, config=self.config)

    def test_cloud_flat(self):
        source_uri = "tiledb://TileDB-Inc/sift_10k"
        queries_uri = "test/data/siftsmall/siftsmall_query.fvecs"
        gt_uri = "test/data/siftsmall/siftsmall_groundtruth.ivecs"
        index_uri = self.flat_index_uri
        k = 100
        nqueries = 100

        query_vectors = load_fvecs(queries_uri)
        gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

        index = vs.ingest(
            index_type="FLAT",
            index_uri=index_uri,
            source_uri=source_uri,
            config=self.config_dict,
            mode=Mode.BATCH,
        )
        _, result_i = index.query(query_vectors, k=k)
        assert accuracy(result_i, gt_i) > MINIMUM_ACCURACY

    def test_cloud_ivf_flat(self):
        source_uri = "tiledb://TileDB-Inc/sift_10k"
        queries_uri = "test/data/siftsmall/siftsmall_query.fvecs"
        gt_uri = "test/data/siftsmall/siftsmall_groundtruth.ivecs"
        index_uri = self.ivf_flat_index_uri
        k = 100
        partitions = 100
        nqueries = 100
        nprobe = 20

        query_vectors = load_fvecs(queries_uri)
        gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

        index = vs.ingest(
            index_type="IVF_FLAT",
            index_uri=index_uri,
            source_uri=source_uri,
            partitions=partitions,
            input_vectors_per_work_item=5000,
            config=self.config_dict,
            mode=Mode.BATCH,
        )
        _, result_i = index.query(query_vectors, k=k, nprobe=nprobe)
        assert accuracy(result_i, gt_i) > MINIMUM_ACCURACY

        _, result_i = index.query(query_vectors, k=k, nprobe=nprobe, mode=Mode.REALTIME, num_partitions=2)
        assert accuracy(result_i, gt_i) > MINIMUM_ACCURACY
