import tiledb.vector_search as vs
import tiledb
import numpy as np

from common import *

def test_load_matrix(tmpdir):
  p = str(tmpdir.mkdir("test").join("test.tdb"))
  data = np.random.rand(9).astype(np.float32).reshape(3,3)

  # write some test data with tiledb-py
  create_array(p, data)

  # load the data with the vector search API and compare
  m, orig_matrix = vs.load_as_array(p, return_matrix=True)
  assert np.array_equal(m, data)

  # mutate and compare again - should match backing data in the C++ Matrix
  data[0,0] = np.random.rand(1).astype(np.float32)
  m[0,0] = data[0,0]
  assert np.array_equal(m, data)
  assert np.array_equal(orig_matrix[0,0], data[0,0])