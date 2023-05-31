import tiledb
import tiledbvspy

import numpy as np

from common import *

def test_tdbMatrix(tmpdir):
  p = str(tmpdir.mkdir("test").join("test.tdb"))
  data = np.random.rand(9).astype(np.float32).reshape(3,3)

  create_array(p, data)

  m = tiledbvspy.tdbMatrix_f32(p)
  m_array = np.array(m)
  assert m_array.shape == data.shape
  assert np.array_equal(m_array, data)

  m_array2 = np.array(m, copy=False) # mutable view
  v = np.random.rand(1).astype(np.float32)
  m_array2[1,1] = v
  data[1,1] = v
  assert np.array_equal(m_array2, data)
  assert m[1,1]
