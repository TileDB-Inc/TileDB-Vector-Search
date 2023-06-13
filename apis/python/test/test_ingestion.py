import numpy as np
from tiledb.vector_search.ingestion import ingest

from common import *


def test_local_ingestion():
  dataset_dir = "/tmp/test-dataset"
  create_random_dataset(nb=10000, d=1000, nq=100, k=10, path=dataset_dir)
  array_uri = "/tmp/test-ingestion"
  source_type = "F32BIN"
  dtype = np.float32
  k = 10

  query_vectors = get_queries(dataset_dir, dtype=dtype)
  gt_i, gt_d = get_groundtruth(dataset_dir, k)

  index = ingest(index_type="FLAT",
                 array_uri=array_uri,
                 source_uri=dataset_dir + "/data",
                 source_type=source_type)
  result = np.transpose(index.query(np.transpose(query_vectors), k=k))
  assert (np.array_equal(result, gt_i))
