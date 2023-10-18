import numpy as np

from common import *

from tiledb.vector_search.index import Index
from tiledb.vector_search import flat_index
from tiledb.vector_search import ivf_flat_index


def test_flat_index(tmp_path):
    uri = os.path.join(tmp_path, "array")
    index = flat_index.create(uri=uri, dimensions=3, vector_type=np.dtype(np.uint8))
    _, result = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([0, 0, 0], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[2] = np.array([2, 2, 2], dtype=np.dtype(np.uint8))
    update_vectors[3] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    update_vectors[4] = np.array([4, 4, 4], dtype=np.dtype(np.uint8))
    index.update_batch(
        vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert 2 in result_i
    assert 3 in result_i
    assert 1 in result_i

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert 2 in result_i
    assert 3 in result_i
    assert 1 in result_i

    index.delete_batch(external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert 2 in result_i
    assert 4 in result_i
    assert 0 in result_i

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert 2 in result_i
    assert 4 in result_i
    assert 0 in result_i

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    index.update_batch(
        vectors=update_vectors, external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert 2 in result_i
    assert 3 in result_i
    assert 1 in result_i

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert 2 in result_i
    assert 3 in result_i
    assert 1 in result_i

    index.delete_batch(external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert 2 in result_i
    assert 4 in result_i
    assert 0 in result_i

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
    assert 2 in result_i
    assert 4 in result_i
    assert 0 in result_i


def test_ivf_flat_index(tmp_path):
    partitions = 10
    uri = os.path.join(tmp_path, "array")
    index = ivf_flat_index.create(uri=uri, dimensions=3, vector_type=np.dtype(np.uint8), partitions=partitions)
    _, result = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([0, 0, 0], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[2] = np.array([2, 2, 2], dtype=np.dtype(np.uint8))
    update_vectors[3] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    update_vectors[4] = np.array([4, 4, 4], dtype=np.dtype(np.uint8))
    index.update_batch(
        vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions)
    assert 2 in result_i
    assert 3 in result_i
    assert 1 in result_i

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions)
    assert 2 in result_i
    assert 3 in result_i
    assert 1 in result_i

    index.delete_batch(external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions)
    assert 2 in result_i
    assert 4 in result_i
    assert 0 in result_i

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions)
    assert 2 in result_i
    assert 4 in result_i
    assert 0 in result_i

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    index.update_batch(
        vectors=update_vectors, external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions)
    assert 2 in result_i
    assert 3 in result_i
    assert 1 in result_i

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions)
    assert 2 in result_i
    assert 3 in result_i
    assert 1 in result_i

    index.delete_batch(external_ids=np.array([1, 3]))
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions)
    assert 2 in result_i
    assert 4 in result_i
    assert 0 in result_i

    index = index.consolidate_updates()
    result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3, nprobe=partitions)
    assert 2 in result_i
    assert 4 in result_i
    assert 0 in result_i