import tiledb
import numpy as np
import io

def _load_vecs_t(uri, dtype, ctx_or_config=None):
  with tiledb.scope_ctx(ctx_or_config) as ctx:
      dtype = np.dtype(dtype)
      vfs = tiledb.VFS(ctx.config())
      with vfs.open(uri, "rb") as f:
          d = f.read(-1)
          raw = np.frombuffer(d, dtype=np.uint8)
          ndim = raw[:4].view(np.int32)[0]

          elem_nbytes = int(4 + ndim * dtype.itemsize)
          if raw.size % elem_nbytes != 0:
              raise ValueError(
                  f"Mismatched dims to bytes in file {uri}: {raw.size}, elem_nbytes"
              )
          # take a view on the whole array as
          # (ndim, sizeof(t)*ndim), and return the actual elements
          #return raw.view(np.uint8).reshape((elem_nbytes,-1))[4:,:].view(dtype).reshape((ndim,-1))

          if dtype != np.uint8:
              return raw.view(np.int32).reshape((-1,ndim + 1))[:,1:].view(dtype)
          else:
              return raw.view(np.uint8).reshape((-1,ndim + 1))[:,1:].view(dtype)
          #return raw

def load_ivecs(uri, ctx_or_config=None):
  return _load_vecs_t(uri, np.int32, ctx_or_config)

def load_fvecs(uri, ctx_or_config=None):
  return _load_vecs_t(uri, np.float32, ctx_or_config)

def load_bvecs(uri, ctx_or_config=None):
  return _load_vecs_t(uri, np.uint8, ctx_or_config)