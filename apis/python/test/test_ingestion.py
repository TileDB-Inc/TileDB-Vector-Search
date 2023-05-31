import os

from tiledb.vector_search.ingestion import ingest

config = {"vfs.s3.aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
          "vfs.s3.aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY")}

million = 1000000
ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion-1",
       source_uri="s3://tiledb-nikos/vector-search/datasets/base.1B.u8bin",
       source_type="U8BIN",
       size=100 * million,
       training_sample_size=1 * million,
       config=config,
       verbose=True)

# ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion",
#        source_uri="s3://tiledb-nikos/vector-search/datasets/base.1B.u8bin",
#        source_type="U8BIN",
#        copy_centroids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/centroids.tdb",
#        size=100000,
#        partitions=1000,
#        config=config)

# ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion",
#        source_uri="s3://tiledb-nikos/vector-search/datasets/base.1B.u8bin",
#        source_type="U8BIN",
#        size=1000000,
#        config=config)
#
# ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion",
#        source_uri="s3://tiledb-nikos/vector-search/datasets/base.1B.u8bin",
#        source_type="U8BIN",
#        size=10000000,
#        config=config)
#
# ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion",
#        source_uri="s3://tiledb-nikos/vector-search/datasets/base.1B.u8bin",
#        source_type="U8BIN",
#        size=100000000,
#        config=config)
#
# ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion",
#        source_uri="s3://tiledb-nikos/vector-search/datasets/base.1B.u8bin",
#        source_type="U8BIN",
#        config=config)
#
# ingest(array_uri="s3://tiledb-nikos/vector-search/test-ingestion",
#        source_uri="s3://tiledb-andrew/sift/sift_base",
#        source_type=SourceType.TILEDB_ARRAY)
