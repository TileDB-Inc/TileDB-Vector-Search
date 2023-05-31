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

  m[1,1] = data[1,1] = np.random.rand(1).astype(np.float32)
  assert np.array_equal(m_array, data)
