import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np

import tiledb
from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search._tiledbvspy import *


def load_as_matrix(
    path: str,
    ctx: "Ctx" = None,
    config: Optional[Mapping[str, Any]] = None,
    size: Optional[int] = None,
    timestamp: int = 0,
):
    """
    Load array as Matrix class. We read in all rows (i.e. from 0 to the row domain length).

    Parameters
    ----------
    path: str
        Array path
    ctx: vspy.Ctx
        The vspy Context
    size: int
        Size of vectors to load. If not set we will read from 0 to the column domain length.
    """
    # If the user passes a tiledb python Config object convert to a dictionary
    if isinstance(config, tiledb.Config):
        config = dict(config)

    if ctx is None:
        ctx = vspy.Ctx(config)

    a = tiledb.ArraySchema.load(path, ctx=tiledb.Ctx(config))
    dtype = a.attr(0).dtype
    # Read all rows from column 0 -> `size`. Set no upper_bound. Note that if `size` is None then
    # we'll read to the column domain length.
    if dtype == np.float32:
        m = tdbColMajorMatrix_f32(ctx, path, 0, None, 0, size, 0, timestamp)
    elif dtype == np.float64:
        m = tdbColMajorMatrix_f64(ctx, path, 0, None, 0, size, 0, timestamp)
    elif dtype == np.int32:
        m = tdbColMajorMatrix_i32(ctx, path, 0, None, 0, size, 0, timestamp)
    elif dtype == np.int32:
        m = tdbColMajorMatrix_i64(ctx, path, 0, None, 0, size, 0, timestamp)
    elif dtype == np.uint8:
        m = tdbColMajorMatrix_u8(ctx, path, 0, None, 0, size, 0, timestamp)
    elif dtype == np.int8:
        m = tdbColMajorMatrix_i8(ctx, path, 0, None, 0, size, 0, timestamp)
    # elif dtype == np.uint64:
    #     return tdbColMajorMatrix_u64(ctx, path, size, timestamp)
    else:
        raise ValueError("Unsupported Matrix dtype: {}".format(a.attr(0).dtype))
    m.load()
    return m


def load_as_array(
    path,
    return_matrix: bool = False,
    ctx: "Ctx" = None,
    config: Optional[Mapping[str, Any]] = None,
    size: Optional[int] = None,
):
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
    m = load_as_matrix(path, size=size, ctx=ctx, config=config)
    r = np.array(m, copy=False)

    # hang on to a copy for testing purposes, for now
    if return_matrix:
        return r, m
    else:
        return r


def debug_slice(m: "colMajorMatrix", name: str):
    dtype = m.dtype
    if dtype == np.float32:
        return debug_matrix_f32(m, name)
    elif dtype == np.uint8:
        return debug_matrix_u8(m, name)
    elif dtype == np.int8:
        return debug_matrix_i8(m, name)
    elif dtype == np.uint64:
        return debug_matrix_u64(m, name)
    else:
        raise TypeError(f"Unsupported type: {dtype}!")


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
    elif db.dtype == np.int8:
        return query_vq_i8(db, *args)
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
    elif db.dtype == np.int8:
        return vq_query_heap_i8(db, *args)
    else:
        raise TypeError("Unknown type!")


def query_vq_heap_pyarray(db: "colMajorMatrix", *args):
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
        return vq_query_heap_pyarray_f32(db, *args)
    elif db.dtype == np.uint8:
        return vq_query_heap_pyarray_u8(db, *args)
    elif db.dtype == np.int8:
        return vq_query_heap_pyarray_i8(db, *args)
    else:
        raise TypeError("Unknown type!")


def ivf_index_tdb(
    dtype: np.dtype,
    db_uri: str,
    external_ids_uri: str,
    deleted_ids: "Vector",
    centroids_uri: str,
    parts_uri: str,
    index_array_uri: str,
    id_uri: str,
    start: int = 0,
    end: int = 0,
    nthreads: int = 0,
    timestamp: int = 0,
    config: Dict = None,
):
    if config is None:
        ctx = vspy.Ctx({})
    else:
        ctx = vspy.Ctx(config)

    args = tuple(
        [
            ctx,
            db_uri,
            external_ids_uri,
            deleted_ids,
            centroids_uri,
            parts_uri,
            index_array_uri,
            id_uri,
            start,
            end,
            nthreads,
            timestamp,
        ]
    )

    if dtype == np.float32:
        return ivf_index_tdb_f32(*args)
    elif dtype == np.uint8:
        return ivf_index_tdb_u8(*args)
    elif dtype == np.int8:
        return ivf_index_tdb_i8(*args)
    else:
        raise TypeError("Unknown type!")


def ivf_index(
    dtype: np.dtype,
    db: "colMajorMatrix",
    external_ids: "Vector",
    deleted_ids: "Vector",
    centroids_uri: str,
    parts_uri: str,
    index_array_uri: str,
    id_uri: str,
    start: int = 0,
    end: int = 0,
    nthreads: int = 0,
    timestamp: int = 0,
    config: Dict = None,
):
    if config is None:
        ctx = vspy.Ctx({})
    else:
        ctx = vspy.Ctx(config)

    args = tuple(
        [
            ctx,
            db,
            external_ids,
            deleted_ids,
            centroids_uri,
            parts_uri,
            index_array_uri,
            id_uri,
            start,
            end,
            nthreads,
            timestamp,
        ]
    )

    if dtype == np.float32:
        return ivf_index_f32(*args)
    elif dtype == np.uint8:
        return ivf_index_u8(*args)
    elif dtype == np.int8:
        return ivf_index_i8(*args)
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
    nthreads: int
        Number of theads
    ctx: vspy.Ctx
        The vspy Context
    """
    if ctx is None:
        ctx = vspy.Ctx({})

    args = tuple(
        [
            parts_db,
            centroids_db,
            query_vectors,
            indices,
            ids,
            nprobe,
            k_nn,
            nthreads,
        ]
    )

    # logging.info(f">>>> ivf_query_ram len(indices): {len(indices)}, dtype: {dtype}, use_nuv_implementation: {use_nuv_implementation}")
    # pdb.set_trace()

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
    elif dtype == np.int8:
        if use_nuv_implementation:
            return nuv_query_heap_infinite_ram_reg_blocked_i8(*args)
        else:
            return qv_query_heap_infinite_ram_i8(*args)
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
    nthreads: int,
    ctx: "Ctx" = None,
    use_nuv_implementation: bool = False,
    timestamp: int = 0,
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
    nthreads: int
        Number of theads
    ctx: vspy.Ctx
        The vspy Context
    timestamp: int
        Read timestamp
    """
    if ctx is None:
        ctx = vspy.Ctx({})

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
            nthreads,
            timestamp,
        ]
    )

    logging.info(
        f">>>> module.py: ivf_query_ram len(indices): {len(indices)}, dtype: {dtype}, use_nuv_implementation: {use_nuv_implementation}"
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
    elif dtype == np.int8:
        if use_nuv_implementation:
            return nuv_query_heap_finite_ram_reg_blocked_i8(*args)
        else:
            return qv_query_heap_finite_ram_i8(*args)
    else:
        raise TypeError("Unknown type!")


def partition_ivf_index(centroids, query, nprobe=1, nthreads=0):
    if query.dtype == np.float32:
        return partition_ivf_index_f32(centroids, query, nprobe, nthreads)
    elif query.dtype == np.uint8:
        return partition_ivf_index_u8(centroids, query, nprobe, nthreads)
    elif query.dtype == np.int8:
        return partition_ivf_index_i8(centroids, query, nprobe, nthreads)
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
    timestamp: int = 0,
):
    if ctx is None:
        ctx = vspy.Ctx({})
    args = tuple(
        [
            ctx,  # 0
            parts_uri,  # 1
            StdVector_u64(active_partitions),  # 2
            query_vectors,  # 3
            active_queries,  # 4
            StdVector_u64(indices),
            ids_uri,
            k_nn,
            timestamp,
        ]
    )

    # logging.info(f">>>> ivf_query_ram len(indices): {len(indices)}, dtype: {dtype},")
    # pdb.set_trace()

    if dtype == np.float32:
        return dist_qv_f32(*args)
    elif dtype == np.uint8:
        return dist_qv_u8(*args)
    elif dtype == np.int8:
        return dist_qv_i8(*args)
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
    elif array.dtype == np.int8:
        return pyarray_copyto_matrix_i8(array)
    else:
        raise TypeError("Unsupported type!")


def kmeans_fit(
    partitions: int,
    init: str,
    max_iter: int,
    verbose: bool,
    n_init: int,
    sample_vectors: "colMajorMatrix",
    tol: Optional[float] = None,
    nthreads: Optional[int] = None,
    seed: Optional[int] = None,
):
    args = tuple(
        [
            partitions,
            init,
            max_iter,
            verbose,
            n_init,
            sample_vectors,
            tol,
            nthreads,
            seed,
        ]
    )
    if sample_vectors.dtype == np.float32:
        return kmeans_fit_f32(*args)
    else:
        raise TypeError("Unsupported type!")


def kmeans_predict(centroids: "colMajorMatrix", sample_vectors: "colMajorMatrix"):
    args = tuple(
        [
            centroids,
            sample_vectors,
        ]
    )
    if sample_vectors.dtype == np.float32:
        return kmeans_predict_f32(*args)
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
