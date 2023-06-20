from typing import Dict

import numpy as np

import tiledb
from tiledb.vector_search._tiledbvspy import *
from tiledb.vector_search import _tiledbvspy as cc


def load_as_matrix(path: str, nqueries: int = 0, config: Dict = {}):
    """
    Load array as Matrix class

    Parameters
    ----------
    path: str
        Array path
    nqueries: int
        Number of queries
    config: Dict
        TileDB configuration parameters
    """
    ctx = Ctx(config)

    a = tiledb.ArraySchema.load(path)
    dtype = a.attr(0).dtype
    if dtype == np.float32:
        return tdbColMajorMatrix_f32(ctx, path, nqueries)
    elif dtype == np.float64:
        return tdbColMajorMatrix_f64(ctx, path, nqueries)
    elif dtype == np.int32:
        return tdbColMajorMatrix_i32(ctx, path, nqueries)
    elif dtype == np.int32:
        return tdbColMajorMatrix_i64(ctx, path, nqueries)
    elif dtype == np.uint8:
        return tdbColMajorMatrix_u8(ctx, path, nqueries)
    # elif dtype == np.uint64:
    #     return tdbColMajorMatrix_u64(ctx, path, nqueries)
    else:
        raise ValueError("Unsupported Matrix dtype: {}".format(a.attr(0).dtype))


def load_as_array(path, return_matrix: bool = False, config: Dict = {}):
    """
    Load array as array class

    Parameters
    ----------
    path: str
        Array path
    return_matrix: bool
        Return matrix
    config: Dict
        TileDB configuration parameters
    """
    m = load_as_matrix(path, config=config)
    r = np.array(m, copy=False)

    # hang on to a copy for testing purposes, for now
    if return_matrix:
        return r, m
    else:
        return r


def query_vq(db: "colMajorMatrix", *args):
    """
    Run vector query

    Parameters
    ----------
    db: colMajorMatrix
        Open Matrix class from load_as_matrix
    args:
        Args for query
    """
    if db.dtype == np.float32:
        return query_vq_f32(db, *args)
    elif db.dtype == np.uint8:
        return query_vq_u8(db, *args)
    else:
        raise TypeError("Unknown type!")


def query_kmeans(
    dtype: np.dtype,
    parts_uri: str,
    centroids_db: "colMajorMatrix",
    query_vectors: "colMajorMatrix",
    index_db: "Vector",
    ids_uri: str,
    nprobe: int,
    k_nn: int,
    nth: bool,
    nthreads: int,
    ctx: "Ctx" = None,
):
    """
    Run kmeans vector query

    Parameters
    ----------
    dtype: numpy.dtype
        Type of vectpr, flaot32 or uint8
    parts_uri: str
        Partition URI
    centroids_db: colMajorMatrix
        Open Matrix class from load_as_matrix for centroids
    query_vectors: colMajorMatrix
        Open Matrix class from load_as_matrix for queries
    index_db: Vector
        Vectors
    ids_uri: str
        URI for id mappings
    nprobe: int
        Number of probs
    k_nn: int
        Number of nn
    nth: bool
        Return nth records
    nthreads: int
        Number of theads
    ctx: Ctx
        Tiledb Context
    """
    if ctx is None:
        ctx = Ctx({})

    args = tuple(
        [
            ctx,
            parts_uri,
            centroids_db,
            query_vectors,
            index_db,
            ids_uri,
            nprobe,
            k_nn,
            nth,
            nthreads,
        ]
    )

    if dtype == np.float32:
        return kmeans_query_f32(*args)
    elif dtype == np.uint8:
        return kmeans_query_u8(*args)
    else:
        raise TypeError("Unknown type!")


def validate_top_k(results: np.ndarray, ground_truth: np.ndarray):
    if results.dtype == np.uint64:
        return cc.validate_top_k_u64(results, ground_truth)
    else:
        raise TypeError("Unknown type for validate_top_k!")


def array_to_matrix(array: np.ndarray):
    if array.dtype == np.float32:
        return pyarray_copyto_matrix_f32(array)
    elif array.dtype == np.float64:
        return pyarray_copyto_matrix_f64(array)
    elif array.dtype == np.uint8:
        return pyarray_copyto_matrix_u8(array)
    elif array.dtype == np.int32:
        return pyarray_copyto_matrix_i32(array)
    elif array.dtype == np.uint64:
        return pyarray_copyto_matrix_u64(array)
    else:
        raise TypeError("Unsupported type!")
