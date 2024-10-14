import io
import json
from typing import Optional

import numpy as np

import tiledb
from tiledb.vector_search import _tiledbvspy as vspy

MAX_INT32 = np.iinfo(np.dtype("int32")).max
MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
MAX_FLOAT32 = np.finfo(np.dtype("float32")).max


def is_type_erased_index(index_type: str) -> bool:
    return index_type == "VAMANA" or index_type == "IVF_PQ"


def create_array_and_add_to_group(array_uri, array_name, schema, group):
    tiledb.Array.create(array_uri, schema)
    add_to_group(group, array_uri, name=array_name)


def add_to_group(group, uri, name):
    """
    Adds an object to a group. Automatically infers whether to use a relative path or absolute path.
    NOTE(paris): We use absolute paths for tileDB URIs because of a bug tracked in SC39197, once
    that is fixed everything can use relative paths.
    """
    if "tiledb://" in uri:
        group.add(uri, name=name)
    else:
        group.add(name, name=name, relative=True)


def to_temporal_policy(timestamp) -> Optional[vspy.TemporalPolicy]:
    temporal_policy = None
    if isinstance(timestamp, tuple):
        if len(timestamp) != 2:
            raise ValueError(
                "'timestamp' argument expects either int or tuple(start: int, end: int)"
            )
        temporal_policy = vspy.TemporalPolicy(timestamp[0], timestamp[1])
    elif timestamp is not None:
        temporal_policy = vspy.TemporalPolicy(timestamp)
    return temporal_policy


def metadata_to_list(group, key):
    return [int(x) for x in list(json.loads(group.meta.get(key, "[]")))]


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
                    f"Mismatched dims to bytes in file {uri}: raw.size: {raw.size}, elem_nbytes: {elem_nbytes}"
                )
            # take a view on the whole array as
            # (ndim, sizeof(t)*ndim), and return the actual elements
            # return raw.view(np.uint8).reshape((elem_nbytes,-1))[4:,:].view(dtype).reshape((ndim,-1))

            if dtype != np.uint8:
                return raw.view(np.int32).reshape((-1, ndim + 1))[:, 1:].view(dtype)
            else:
                return raw.view(np.uint8).reshape((-1, ndim + 1))[:, 1:].view(dtype)
            # return raw


def load_ivecs(uri, ctx_or_config=None):
    return _load_vecs_t(uri, np.int32, ctx_or_config)


def load_fvecs(uri, ctx_or_config=None):
    return _load_vecs_t(uri, np.float32, ctx_or_config)


def load_bvecs(uri, ctx_or_config=None):
    return _load_vecs_t(uri, np.uint8, ctx_or_config)


def _write_vecs_t(uri, data, dtype, ctx_or_config=None):
    with tiledb.scope_ctx(ctx_or_config) as ctx:
        dtype = np.dtype(dtype)
        vfs = tiledb.VFS(ctx.config())
        ndim = data.shape[1]

        buffer = io.BytesIO()

        for vector in data:
            buffer.write(np.array([ndim], dtype=np.int32).tobytes())
            buffer.write(vector.tobytes())

        with vfs.open(uri, "wb") as f:
            f.write(buffer.getvalue())


def write_ivecs(uri, data, ctx_or_config=None):
    _write_vecs_t(uri, data, np.int32, ctx_or_config)


def write_fvecs(uri, data, ctx_or_config=None):
    _write_vecs_t(uri, data, np.float32, ctx_or_config)


def normalize_vector(vector: np.array) -> np.array:
    """
    Normalize a single vector to unit length.

    Args:
    vector (np.array): Input vector to be normalized.

    Returns:
    np.array: Normalized vector.
    """
    if vector.dtype == object:
        return np.array([normalize_vector(v) for v in vector])

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize an array of vectors to unit length while preserving the original array structure.

    Args:
    vectors (np.ndarray): Input array of vectors to be normalized.

    Returns:
    np.ndarray: Array of normalized vectors with the same structure as the input.
    """
    normalized = np.empty_like(vectors)
    for i, v in enumerate(vectors):
        normalized[i] = normalize_vector(v)
    return normalized
