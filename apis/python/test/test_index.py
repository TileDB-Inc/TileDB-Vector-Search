import numpy as np
from common import *

import tiledb.vector_search.index as ind
from tiledb.vector_search import flat_index, ivf_flat_index
from tiledb.vector_search.index import Index

def query_and_check(index, queries, k, expected, **kwargs):
    for _ in range(3):
        result_d, result_i = index.query(queries, k=k, **kwargs)
        assert expected.issubset(set(result_i[0]))

def test_flat_index(tmp_path):
    uri = os.path.join(tmp_path, "array")
    index = flat_index.create(uri=uri, dimensions=3, vector_type=np.dtype(np.uint8))
    for _ in range(2):
        result_d, result_i = index.query(np.array([[2, 2, 2]], dtype=np.float32), k=3)
        assert {ind.MAX_UINT64} == set(result_i[0])

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([0, 0, 0], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[2] = np.array([2, 2, 2], dtype=np.dtype(np.uint8))
    update_vectors[3] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    update_vectors[4] = np.array([4, 4, 4], dtype=np.dtype(np.uint8))
    index.update_batch(vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3})

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3})

    index.delete_batch(external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4})

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4})

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    index.update_batch(vectors=update_vectors, external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3})

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3})

    index.delete_batch(external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4})

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4})


def test_ivf_flat_index(tmp_path):
    partitions = 10
    uri = os.path.join(tmp_path, "array")
    index = ivf_flat_index.create(
        uri=uri, dimensions=3, vector_type=np.dtype(np.uint8), partitions=partitions
    )
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {ind.MAX_UINT64}, nprobe=partitions)

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([0, 0, 0], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[2] = np.array([2, 2, 2], dtype=np.dtype(np.uint8))
    update_vectors[3] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    update_vectors[4] = np.array([4, 4, 4], dtype=np.dtype(np.uint8))
    index.update_batch(vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3}, nprobe=partitions)

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3}, nprobe=partitions)

    index.delete_batch(external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4}, nprobe=partitions)

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4}, nprobe=partitions)

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([1, 1, 1], dtype=np.dtype(np.uint8))
    update_vectors[1] = np.array([3, 3, 3], dtype=np.dtype(np.uint8))
    index.update_batch(vectors=update_vectors, external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3}, nprobe=partitions)

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3}, nprobe=partitions)

    index.delete_batch(external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4}, nprobe=partitions)

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4}, nprobe=partitions)
