import numpy as np
from common import *

import tiledb.vector_search.index as ind
from tiledb.vector_search import flat_index, ivf_flat_index
from tiledb.vector_search.index import Index
from tiledb.vector_search.ingestion import ingest

def test_flat_index(tmp_path):
    uri = os.path.join(tmp_path, "array")
    index = flat_index.create(uri=uri, dimensions=3, vector_type=np.dtype(np.uint8))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {ind.MAX_UINT64} == set(result_i[0])

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([0, 0, 0], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[2] = np.array([2, 2, 2], dtype=np.dtype(np.uint8))
    update_vectors[3] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    update_vectors[4] = np.array([4, 4, 4], dtype=np.dtype(np.uint8))
    index.update_batch(vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {1, 2, 3}.issubset(set(result_i[0]))

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {1, 2, 3}.issubset(set(result_i[0]))

    index.delete_batch(external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {0, 2, 4}.issubset(set(result_i[0]))

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {0, 2, 4}.issubset(set(result_i[0]))

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    index.update_batch(vectors=update_vectors, external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {1, 2, 3}.issubset(set(result_i[0]))

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {1, 2, 3}.issubset(set(result_i[0]))

    index.delete_batch(external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {0, 2, 4}.issubset(set(result_i[0]))

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert {0, 2, 4}.issubset(set(result_i[0]))


def test_ivf_flat_index(tmp_path):
    partitions = 10
    uri = os.path.join(tmp_path, "array")
    index = ivf_flat_index.create(
        uri=uri, dimensions=3, vector_type=np.dtype(np.uint8), partitions=partitions
    )
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {ind.MAX_UINT64} == set(result_i[0])

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([0, 0, 0], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[2] = np.array([2, 2, 2], dtype=np.dtype(np.uint8))
    update_vectors[3] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    update_vectors[4] = np.array([4, 4, 4], dtype=np.dtype(np.uint8))
    index.update_batch(vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {1, 2, 3}.issubset(set(result_i[0]))

    index = index.consolidate_updates()
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {1, 2, 3}.issubset(set(result_i[0]))

    index.delete_batch(external_ids=np.array([1, 3]))
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {0, 2, 4}.issubset(set(result_i[0]))

    index = index.consolidate_updates()
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {0, 2, 4}.issubset(set(result_i[0]))

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    index.update_batch(vectors=update_vectors, external_ids=np.array([1, 3]))
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {1, 2, 3}.issubset(set(result_i[0]))

    index = index.consolidate_updates()
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {1, 2, 3}.issubset(set(result_i[0]))

    index.delete_batch(external_ids=np.array([1, 3]))
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {0, 2, 4}.issubset(set(result_i[0]))

    index = index.consolidate_updates()
    result_d, result_i = index.query(
        np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions
    )
    assert {0, 2, 4}.issubset(set(result_i[0]))

# NOTE(paris): This test is failing on my machine. See: https://tiledb.slack.com/archives/C0537B4V7Q8/p1702312614334709
def test_ivf_flat_failing_query(tmp_path):
    # TODO(paris): Pull this and make_blobs() into a helper in common.py. Left to keep failing unit 
    # test easier to debug.
    from sklearn.datasets import make_blobs

    index_type = "IVF_FLAT"
    n_samples = 10000
    dimensions = 10

    dataset_dir = os.path.join(tmp_path, f"dataset_{index_type}_{dimensions}")
    os.mkdir(dataset_dir)
    X, _ = make_blobs(n_samples=n_samples, n_features=dimensions, centers=3, random_state=1)
    with open(os.path.join(dataset_dir, "data.f32bin"), "wb") as f:
        np.array([n_samples, dimensions], dtype="uint32").tofile(f)
        X.astype("float32").tofile(f)

    index_uri = os.path.join(tmp_path, f"array_{index_type}_{dimensions}")
    index = ingest(index_type=index_type, index_uri=index_uri, source_uri=os.path.join(dataset_dir, "data.f32bin"))
    query_shape = (1, dimensions)
    query = np.random.rand(*query_shape).astype(np.float32)
    result_d, result_i = index.query(query, k=1)