import tiledbvspy as vspy

import tiledb
import numpy as np

def load_as_matrix(path, nqueries=0):
    a = tiledb.ArraySchema.load(path)
    dtype = a.attr(0).dtype
    if dtype == np.float32:
        return vspy.tdbColMajorMatrix_f32(path, nqueries)
    elif dtype == np.float64:
        return vspy.tdbColMajorMatrix_f64(path, nqueries)
    elif dtype == np.int32:
        return vspy.tdbColMajorMatrix_i32(path, nqueries)
    elif dtype == np.int32:
        return vspy.tdbColMajorMatrix_i64(path, nqueries)
    elif dtype == np.u64:
        return vspy.tdbColMajorMatrix_u64(path, nqueries)
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
        raise NotImplementedError("vquery_vq[u8] not implemented")
        #vspy.query_vq_u8(db, **args)
    else:
        raise TypeError("Unknown type!")