from typing import Dict

import numpy as np

import tiledb
from tiledb.vector_search._tiledbvspy import *

from typing import Optional


def load_as_matrix(path: str, nqueries: int = 0, ctx: "Ctx" = None):
    """
    Load array as Matrix class

    Parameters
    ----------
    path: str
        Array path
    nqueries: int
        Number of queries
    ctx: Ctx
        TileDB context
    """
    if ctx is None:
        ctx = Ctx({})

    a = tiledb.ArraySchema.load(path)
    dtype = a.attr(0).dtype
    if dtype == np.float32:
        m = tdbColMajorMatrix_f32(ctx, path, nqueries)
    elif dtype == np.float64:
        m = tdbColMajorMatrix_f64(ctx, path, nqueries)
    elif dtype == np.int32:
        m = tdbColMajorMatrix_i32(ctx, path, nqueries)
    elif dtype == np.int32:
        m = tdbColMajorMatrix_i64(ctx, path, nqueries)
    elif dtype == np.uint8:
        m = tdbColMajorMatrix_u8(ctx, path, nqueries)
    # elif dtype == np.uint64:
    #     return tdbColMajorMatrix_u64(ctx, path, nqueries)
    else:
        raise ValueError("Unsupported Matrix dtype: {}".format(a.attr(0).dtype))
    m.load()
    return m


def load_as_array(path, return_matrix: bool = False, ctx: "Ctx" = None):
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
    m = load_as_matrix(path, ctx=ctx)
    r = np.array(m, copy=False)

    # hang on to a copy for testing purposes, for now
    if return_matrix:
        return r, m
    else:
        return r


def query_vq_nth(db: "colMajorMatrix", *args):
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


def query_vq_heap(db: "colMajorMatrix", *args):
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
        return vq_query_heap_f32(db, *args)
    elif db.dtype == np.uint8:
        return vq_query_heap_u8(db, *args)
    else:
        raise TypeError("Unknown type!")


def ivf_index_tdb(
    dtype: np.dtype,
    db_uri: str,
    centroids_uri: str,
    parts_uri: str,
    index_uri: str,
    id_uri: str,
    start: int = 0,
    end: int = 0,
    nthreads: int = 0,
    config: Dict = None,
):
    if config is None:
        ctx = Ctx({})
    else:
        ctx = Ctx(config)

    args = tuple(
        [ctx, db_uri, centroids_uri, parts_uri, index_uri, id_uri, start, end, nthreads]
    )

    if dtype == np.float32:
        return ivf_index_tdb_f32(*args)
    elif dtype == np.uint8:
        return ivf_index_tdb_u8(*args)
    else:
        raise TypeError("Unknown type!")


def ivf_index(
    dtype: np.dtype,
    db: "colMajorMatrix",
    centroids_uri: str,
    parts_uri: str,
    index_uri: str,
    id_uri: str,
    start: int = 0,
    end: int = 0,
    nthreads: int = 0,
    config: Dict = None,
):
    if config is None:
        ctx = Ctx({})
    else:
        ctx = Ctx(config)

    args = tuple(
        [ctx, db, centroids_uri, parts_uri, index_uri, id_uri, start, end, nthreads]
    )

    if dtype == np.float32:
        return ivf_index_f32(*args)
    elif dtype == np.uint8:
        return ivf_index_u8(*args)
    else:
        raise TypeError("Unknown type!")


def ivf_query_ram(
    dtype: np.dtype,
    parts_db: "colMajorMatrix",
    centroids_db: "colMajorMatrix",
    query_vectors: "colMajorMatrix",
    indices: "Vector",
    ids: "Vector",
    nprobe: int,
    k_nn: int,
    nth: bool,
    nthreads: int,
    ctx: "Ctx" = None,
    use_nuv_implementation: bool = False,
):
    """
    Run IVF vector query using infinite RAM

    Parameters
    ----------
    dtype: numpy.dtype
        Type of vector, float32 or uint8
    parts_db: colMajorMatrix
        Partitioned vectors
    centroids_db: colMajorMatrix
        Open Matrix class from load_as_matrix for centroids
    query_vectors: colMajorMatrix
        Open Matrix class from load_as_matrix for queries
    indices: Vector
        Partition indices
    ids: Vector
        Vector IDs
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
            parts_db,
            centroids_db,
            query_vectors,
            indices,
            ids,
            nprobe,
            k_nn,
            nth,
            nthreads,
        ]
    )

    if dtype == np.float32:
        if use_nuv_implementation:
            return nuv_query_heap_infinite_ram_reg_blocked_f32(*args)
        else:
            return qv_query_heap_infinite_ram_f32(*args)
    elif dtype == np.uint8:
        if use_nuv_implementation:
            return nuv_query_heap_infinite_ram_reg_blocked_u8(*args)
        else:
            return qv_query_heap_infinite_ram_u8(*args)
    else:
        raise TypeError("Unknown type!")


def ivf_query(
    dtype: np.dtype,
    parts_uri: str,
    centroids: "colMajorMatrix",
    query_vectors: "colMajorMatrix",
    indices: "Vector",
    ids_uri: str,
    nprobe: int,
    k_nn: int,
    memory_budget: int,
    nth: bool,
    nthreads: int,
    ctx: "Ctx" = None,
    use_nuv_implementation: bool = False,
):
    """
    Run IVF vector query using a memory budget

    Parameters
    ----------
    dtype: numpy.dtype
        Type of vector, float32 or uint8
    parts_uri: str
        Partition URI
    centroids: colMajorMatrix
        Open Matrix class from load_as_matrix for centroids
    query_vectors: colMajorMatrix
        Open Matrix class from load_as_matrix for queries
    indeces: Vector
        Vectors
    ids_uri: str
        URI for id mappings
    nprobe: int
        Number of probs
    k_nn: int
        Number of nn
    memory_budget: int
        Main memory budget
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
            centroids,
            query_vectors,
            indices,
            ids_uri,
            nprobe,
            k_nn,
            memory_budget,
            nth,
            nthreads,
        ]
    )

    if dtype == np.float32:
        if use_nuv_implementation:
            return nuv_query_heap_finite_ram_reg_blocked_f32(*args)
        else:
            return qv_query_heap_finite_ram_f32(*args)
    elif dtype == np.uint8:
        if use_nuv_implementation:
            return nuv_query_heap_finite_ram_reg_blocked_u8(*args)
        else:
            return qv_query_heap_finite_ram_u8(*args)
    else:
        raise TypeError("Unknown type!")


def partition_ivf_index(centroids, query, nprobe=1, nthreads=0):
    if query.dtype == np.float32:
        return partition_ivf_index_f32(centroids, query, nprobe, nthreads)
    elif query.dtype == np.uint8:
        return partition_ivf_index_u8(centroids, query, nprobe, nthreads)
    else:
        raise TypeError("Unsupported type!")


def dist_qv(
    dtype: np.dtype,
    parts_uri: str,
    ids_uri: str,
    query_vectors: "colMajorMatrix",
    active_partitions: np.array,
    active_queries: np.array,
    indices: np.array,
    k_nn: int,
    ctx: "Ctx" = None,
):
    if ctx is None:
        ctx = Ctx({})
    args = tuple(
        [
            ctx,
            parts_uri,
            active_partitions,
            query_vectors,
            active_queries,
            StdVector_u64(indices),
            ids_uri,
            k_nn,
        ]
    )
    if dtype == np.float32:
        return dist_qv_f32(*args)
    elif dtype == np.uint8:
        return dist_qv_u8(*args)
    else:
        raise TypeError("Unsupported type!")


def validate_top_k(results: np.ndarray, ground_truth: np.ndarray):
    if results.dtype == np.uint64:
        return validate_top_k_u64(results, ground_truth)
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


# TODO
# def load_partitioned(uri, partitions, dtype: Optional[np.dtype] = None):
#    if dtype is None:
#        arr = tiledb.open(uri).dtype
#    if dtype == np.float32:
#        return tdbPartitionedMatrix_f32(uri,
#    elif dtype == np.uint8:
#        return tdbPartitionedMatrix_f32(uri,
#    else:
#        raise TypeError("Unknown type!")

# class PartitionedMatrix:
#    def __init__(self, uri, partitions):
#        self.uri = uri
#        self.partitions = partitions
#
#        self._m = load_partitioned(uri, partitions)
