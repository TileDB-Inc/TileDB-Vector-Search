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

    @classmethod
    def setUpClass(cls):
        token = os.getenv("TILEDB_REST_TOKEN")
        if os.getenv("TILEDB_CLOUD_HELPER_VAR"):
            token = os.getenv("TILEDB_CLOUD_HELPER_VAR")
        tiledb.cloud.login(token = token)
        namespace, storage_path, _ = groups._default_ns_path_cred()
        storage_path = storage_path.replace("//", "/").replace("/", "//", 1)
        rand_name = random_name("vector_search")
        test_path = f"tiledb://{namespace}/{storage_path}/{rand_name}"
        print(test_path)
        cls.ivf_flat_index_uri = f"{test_path}/test_ivf_flat_array"
        cls.ivf_flat_random_sampling_index_uri = (
            f"{test_path}/test_ivf_flat_random_sampling_array"
        )

    @classmethod
    def tearDownClass(cls):
        vs.Index.delete_index(uri=cls.ivf_flat_index_uri, config=tiledb.cloud.Config())

    def test_cloud_ivf_flat(self):
        source_uri = "tiledb://TileDB-Inc/6a9a8e97-d99c-4ddb-829a-8455c794906e"
        queries_uri = siftsmall_query_file
        gt_uri = siftsmall_groundtruth_file
        index_uri = CloudTests.ivf_flat_index_uri
        k = 100
        partitions = 100
        nqueries = 100
        nprobe = 20

        queries = load_fvecs(queries_uri)
        gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

        index = vs.ingest(
            index_type="IVF_FLAT",
            index_uri=index_uri,
            source_uri=source_uri,
            config=tiledb.cloud.Config().dict(),
            mode=Mode.BATCH,
            size=100000,
            training_sample_size=50000,
            verbose=True
        )