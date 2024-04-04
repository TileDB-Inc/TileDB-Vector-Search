import numpy as np
from common import *

import tiledb.vector_search as vs
from tiledb.vector_search import _tiledbvspy as vspy


def test_load_matrix(tmpdir):
    p = str(tmpdir.mkdir("test").join("test.tdb"))
    data = np.random.rand(12).astype(np.float32).reshape(3, 4)

    # write some test data with tiledb-py
    create_array(p, data)

    # load the data with the vector search API and compare
    m, orig_matrix = vs.load_as_array(p, return_matrix=True)
    assert np.array_equal(m, data)

    # mutate and compare again - should match backing data in the C++ Matrix
    data[0, 0] = np.random.rand(1).astype(np.float32)
    m[0, 0] = data[0, 0]
    assert np.array_equal(m, data)
    assert np.array_equal(orig_matrix[0, 0], data[0, 0])


def test_load_matrix_specify_size(tmpdir):
    p = str(tmpdir.mkdir("test").join("test.tdb"))
    data = np.random.rand(12).astype(np.float32).reshape(3, 4)

    # write some test data with tiledb-py
    create_array(p, data)

    # test specifying a size
    m = vs.load_as_array(p, size=data.shape[1])
    assert np.array_equal(m, data)

    # test specifying a smaller size
    m = vs.load_as_array(p, size=2)
    assert np.array_equal(m, data[:, :2])

    # test specifying a size of 0
    m = vs.load_as_array(p, size=0)
    assert m.shape == (3, 0)


def test_vector(tmpdir):
    v = vspy._create_vector_u64()
    assert np.array_equal(np.array(v), np.arange(10))


def test_partition_ivf_index(tmpdir):
    tmpdir.mkdir("test").join("test.tdb")

    # Test: 3x3 identity; swap columns 0 and 1; check assignment matches swap.
    data = np.identity(3).astype(np.float32)
    data_m = vs.array_to_matrix(data)
    query = data.copy()[:, [1, 0, 2]]
    query_m = vs.array_to_matrix(query)

    r = vspy.partition_ivf_index_f32(data_m, query_m, 1, 2)
    r_a = np.array(r[1], copy=True)

    assert np.array_equal(r_a, np.array([[1], [0], [2]], dtype=np.uint64))


def test_partition_ivf_index2(tmpdir):
    tmpdir.mkdir("test").join("test.tdb")

    # Test: 3x3 identity; swap columns 0 and 1; check assignment matches swap.
    data = np.identity(3).astype(np.float32)
    np.repeat(data, 3)
    data_m = vs.array_to_matrix(data)
    query = np.identity(3).astype(np.float32)[:, [1, 0, 2]]
    query_m = vs.array_to_matrix(query)

    r = vspy.partition_ivf_index_f32(data_m, query_m, 1, 2)
    r_a = np.array(r[1], copy=True)

    assert np.array_equal(r_a, np.array([[1], [0], [2]], dtype=np.uint64))
