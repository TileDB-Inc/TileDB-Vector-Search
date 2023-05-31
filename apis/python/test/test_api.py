import tiledb.vector_search as vs
import tiledb
import numpy as np

from common import *

def test_load_matrix(tmpdir):
  p = str(tmpdir.mkdir("test").join("test.tdb"))
  data = np.random.rand(9).astype(np.float32).reshape(3,3)

  create_array(p, data)

  m = vs.load_as_array(p)
  assert np.array_equal(m, data)