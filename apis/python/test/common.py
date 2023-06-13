import os

import numpy as np
import tiledb


def xbin_mmap(fname, dtype):
  n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
  assert os.stat(fname).st_size == 8 + n * d * np.dtype(dtype).itemsize
  return np.memmap(fname, dtype=dtype, mode="r", offset=8, shape=(n, d))


def get_queries(fname, dtype, nqueries=None):
  x = xbin_mmap(fname, dtype=dtype)
  if nqueries is not None:
    return x.astype(np.float32)[:nqueries, :]
  else:
    return x.astype(np.float32)


def get_groundtruth(fname, k=None, nqueries=None):
  I, D = groundtruth_read(fname, nqueries)
  if k is not None:
    assert k <= 100
    I = I[:, :k]
    D = D[:, :k]
  return I, D


def groundtruth_read(fname, nqueries=None):
  n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
  assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
  f = open(fname, "rb")
  f.seek(4 + 4)
  I = np.fromfile(f, dtype="uint32", count=n * d).reshape(n, d).astype("uint64")
  D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)

  if nqueries is not None:
    return I[:nqueries, :], D[:nqueries, :]
  else:
    return I, D


def create_schema():
  schema = tiledb.ArraySchema(
    domain=tiledb.Domain(*[
      tiledb.Dim(name='__dim_0', domain=(0, 2), tile=3, dtype='int32'),
      tiledb.Dim(name='__dim_1', domain=(0, 3), tile=3, dtype='int32'),
    ]),
    attrs=[
      tiledb.Attr(name='', dtype='float32', var=False, nullable=False),
    ],
    cell_order='col-major',
    tile_order='col-major',
    capacity=10000,
    sparse=False,
  )
  return schema


def create_array(path: str, data):
  schema = create_schema()
  tiledb.Array.create(path, schema)
  with tiledb.open(path, 'w') as A:
    A[:] = data
