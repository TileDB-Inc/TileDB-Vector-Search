import numpy as np
from tiledb import VFS
from tiledb.cloud.dag import Mode
from tiledb.vector_search.index import FlatIndex
from tiledb.vector_search.ingestion import ingest


def test_local_ingestion():
    group_uri = "/tmp/test-ingestion"
    ingest(index_type="FLAT",
           array_uri=group_uri,
           source_uri="test/data/data_10000_20",
           source_type="U8BIN",
           size=10000,
           training_sample_size=10000,
           verbose=True,
           mode=Mode.LOCAL)

    vfs = VFS()
    nqueries = 10
    k = 10
    with vfs.open("test/data/queries_1000_20", "rb") as f:
        f.seek(8)
        query_vectors = np.reshape(
            np.frombuffer(
                f.read(nqueries * 20),
                count=nqueries * 20,
                dtype=np.uint8
            ).astype(np.float32), (nqueries, 20)
        )

    with vfs.open("test/data/gt_10000_1000_20", "rb") as f:
        f.seek(8)
        gt = np.reshape(
            np.frombuffer(
                f.read(nqueries * k * 4),
                count=nqueries * k,
                dtype=np.uint32
            ).astype(np.uint32), (nqueries, k)
        )
        print(gt[0:10])

    index = FlatIndex(group_uri + "/parts.tdb")
    result = index.query(np.transpose(query_vectors), nqueries=nqueries, k=k)
    print(result)

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
