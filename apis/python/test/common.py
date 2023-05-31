import tiledb
import numpy as np

def create_schema():
  schema = tiledb.ArraySchema(
    domain=tiledb.Domain(*[
      tiledb.Dim(name='__dim_0', domain=(0, 2), tile=3, dtype='int32'),
      tiledb.Dim(name='__dim_1', domain=(0, 2), tile=3, dtype='int32'),
    ]),
    attrs=[
      tiledb.Attr(name='', dtype='float32', var=False, nullable=False),
    ],
    cell_order='row-major',
    tile_order='row-major',
    capacity=10000,
    sparse=False,
  )
  return schema

def create_array(path: str, data):
  schema = create_schema()
  tiledb.Array.create(path, schema)
  with tiledb.open(path, 'w') as A:
    A[:] = data