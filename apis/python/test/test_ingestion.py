import os

from tiledb.cloud.dag import Mode
from tiledb.vector_search.ingestion import ingest


def test_local_ingestion():
    ingest(array_uri="/tmp/test-ingestion",
           source_uri="base.1k.u8bin",
           source_type="U8BIN",
           size=1000,
           training_sample_size=1000,
           verbose=True,
           mode=Mode.LOCAL)


def test_distributed_ingestion():
    config = {"vfs.s3.aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
              "vfs.s3.aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY")}
    ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion",
           source_uri="s3://tiledb-nikos/vector-search/datasets/base.1B.u8bin",
           source_type="U8BIN",
           size=1000000,
           training_sample_size=100000,
           config=config,
           verbose=True,
           mode=Mode.BATCH)
