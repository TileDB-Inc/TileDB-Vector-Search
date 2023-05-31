import tiledbvspy as vspy

import tiledb
import numpy as np

def load_as_matrix(path):
  a = tiledb.ArraySchema.load(path)
  if a.attr(0).dtype == np.float32:
    return vspy.tdbMatrix_f32(path)
  elif a.attr(0).dtype == np.float64:
    return vspy.tdbMatrix_f64(path)
  else:
    raise ValueError("Unsupported Matrix dtype: {}".format(a.attr(0).dtype))

def load_as_array(path):
  return np.array(
    load_as_matrix(path), copy=False
  )