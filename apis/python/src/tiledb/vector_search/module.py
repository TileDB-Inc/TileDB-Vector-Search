import numpy as np
import tiledb
from tiledb.vector_search import _tiledbvspy as vspy


def load_as_matrix(path, nqueries=0, config={}):
    ctx = vspy.Ctx({})

    a = tiledb.ArraySchema.load(path)
    dtype = a.attr(0).dtype
    if dtype == np.float32:
        return vspy.tdbColMajorMatrix_f32(ctx, path, nqueries)
    elif dtype == np.float64:
        return vspy.tdbColMajorMatrix_f64(ctx, path, nqueries)
    elif dtype == np.int32:
        return vspy.tdbColMajorMatrix_i32(ctx, path, nqueries)
    elif dtype == np.int32:
        return vspy.tdbColMajorMatrix_i64(ctx, path, nqueries)
    elif dtype == np.uint8:
        return vspy.tdbColMajorMatrix_u8(ctx, path, nqueries)
    elif dtype == np.uint64:
        return vspy.tdbColMajorMatrix_u64(ctx, path, nqueries)
    else:
        raise ValueError("Unsupported Matrix dtype: {}".format(a.attr(0).dtype))


def load_as_array(path, return_matrix=False):
    m = load_as_matrix(path)
    r = np.array(m, copy=False)

    # hang on to a copy for testing purposes, for now
    if return_matrix:
        return r, m
    else:
        return r


def query_vq(db: np.ndarray, *args):
    if db.dtype == np.float32:
        return vspy.query_vq_f32(db, *args)
    elif db.dtype == np.uint8:
        return vspy.query_vq_u8(db, *args)
    else:
        raise TypeError("Unknown type!")
