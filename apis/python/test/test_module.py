import tiledb
import tiledbvspy as vspy

import numpy as np

from common import *

def test_tdbMatrix(tmpdir):
    p = str(tmpdir.mkdir("test").join("test.tdb"))
    data = np.random.rand(12).astype(np.float32).reshape(3,4)

    create_array(p, data)

    m = tiledbvspy.tdbColMajorMatrix_f32(p, 0)
    m_array = np.array(m)
    assert m_array.shape == data.shape
    assert np.array_equal(m_array, data)

    m_array2 = np.array(m, copy=False) # mutable view
    v = np.random.rand(1).astype(np.float32)
    m_array2[1,2] = v
    data[1,2] = v
    assert np.array_equal(m_array2, data)
    assert m[1,2] == v

def test_context(tmpdir):
    p = str(tmpdir.mkdir("test").join("test.tdb"))

    ctx = vspy.Ctx({})
    ctx = vspy.Ctx({"vfs.s3.region": "us-east-1"})