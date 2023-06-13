import numpy as np
from tiledb.vector_search.ingestion import ingest

from common import *


def test_local_ingestion():
  array_uri = "/tmp/test-ingestion"
  source_uri = "test/data/data_10000_20"
  queries_uri = "test/data/queries_1000_20"
  gt_uri = "test/data/gt_10000_1000_20"
  source_type = "F32BIN"
  dtype = np.float32
  k = 10

  query_vectors = get_queries(queries_uri, dtype=dtype)
  gt_i, gt_d = get_groundtruth(gt_uri, k)
  
  index = ingest(index_type="FLAT",
                 array_uri=array_uri,
                 source_uri=source_uri,
                 source_type=source_type)
  result = np.transpose(index.query(np.transpose(query_vectors), k=k))
  assert (np.array_equal(result, gt_i))

# def test_distributed_ingestion():
#     config = {"vfs.s3.aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
#               "vfs.s3.aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY")}
#     ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion",
#            source_uri="s3://tiledb-nikos/vector-search/datasets/base.1B.u8bin",
#            source_type="U8BIN",
#            size=1000000,
#            training_sample_size=100000,
#            config=config,
#            verbose=True,
#            mode=Mode.BATCH)
