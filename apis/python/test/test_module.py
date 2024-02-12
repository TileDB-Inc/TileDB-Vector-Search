import numpy as np
from common import *

import tiledb.vector_search as vs
from tiledb.vector_search import _tiledbvspy as vspy


def test_tdbMatrix(tmpdir):
    d = tmpdir.mkdir("test")
    p = str(d.join("test.tdb"))
    data = np.random.rand(12).astype(np.float32).reshape(3, 4)

    create_array(p, data)

    ctx = vspy.Ctx({})
    m = vspy.tdbColMajorMatrix_f32(ctx, p, 0, 0, 0, 0, 0, 0)
    m.load()
    m_array = np.array(m)
    assert m_array.shape == data.shape
    assert np.array_equal(m_array, data)

    m_array2 = np.array(m, copy=False)  # mutable view
    v = np.random.rand(1).astype(np.float32)
    m_array2[1, 2] = v

    data[1, 2] = v

    assert np.array_equal(m_array2, data)
    assert m[1, 2] == v


def test_array_to_matrix(tmpdir):
    str(tmpdir.mkdir("test").join("test.tdb"))

    data = np.random.rand(12).astype(np.float32).reshape(3, 4)

    mat = vs.array_to_matrix(data)
    mat_view = np.array(mat, copy=True)  # mutable view
    assert np.array_equal(mat_view, data)


def test_context(tmpdir):
    str(tmpdir.mkdir("test").join("test.tdb"))

    vspy.Ctx({})
    vspy.Ctx({"vfs.s3.region": "us-east-1"})
