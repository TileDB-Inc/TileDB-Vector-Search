import json
import os
import random
import shutil
import string

import numpy as np

import tiledb
from tiledb.cloud import groups
from tiledb.vector_search.flat_index import FlatIndex
from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
from tiledb.vector_search.ivf_pq_index import IVFPQIndex
from tiledb.vector_search.storage_formats import STORAGE_VERSION
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.vamana_index import VamanaIndex

INDEXES = ["FLAT", "IVF_FLAT", "VAMANA", "IVF_PQ"]
INDEX_CLASSES = [FlatIndex, IVFFlatIndex, VamanaIndex, IVFPQIndex]
INDEX_FILES = [
    tiledb.vector_search.flat_index,
    tiledb.vector_search.ivf_flat_index,
    tiledb.vector_search.vamana_index,
    tiledb.vector_search.ivf_pq_index,
]


def xbin_mmap(fname, dtype):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * np.dtype(dtype).itemsize
    return np.memmap(fname, dtype=dtype, mode="r", offset=8, shape=(n, d))


def get_groundtruth_ivec(file, k=None, nqueries=None):
    vfs = tiledb.VFS()
    vector_values = 1 + k
    vector_size = vector_values * 4
    read_size = nqueries
    read_offset = 0
    with vfs.open(file, "rb") as f:
        f.seek(read_offset)
        return (
            np.delete(
                np.reshape(
                    np.frombuffer(
                        f.read(read_size * vector_size),
                        count=read_size * vector_values,
                        dtype=np.int32,
                    ).astype(np.int32),
                    (read_size, k + 1),
                ),
                0,
                axis=1,
            ),
            None,
        )


def get_queries(dataset_dir, dtype, nqueries=None):
    fname = os.path.join(dataset_dir, "queries")
    x = xbin_mmap(fname, dtype=dtype)
    if nqueries is not None:
        return x.astype(np.float32)[:nqueries, :]
    else:
        return x.astype(np.float32)


def get_groundtruth(dataset_dir, k=None, nqueries=None):
    I, D = groundtruth_read(dataset_dir, nqueries)
    if k is not None:
        assert k <= 100
        I = I[:, :k]
        D = D[:, :k]
    return I, D


def groundtruth_read(dataset_dir, nqueries=None):
    fname = os.path.join(dataset_dir, "gt")
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


def create_random_dataset_f32_only_data(nb, d, centers, path):
    """
    Creates a random float32 dataset containing just a dataset and then writes it to disk.

    Parameters
    ----------
    nb: int
        Number of points in the dataset
    d: int
        Dimension of the dataset
    nq: int
        Number of centers
    path: str
        Path to write the dataset to
    """
    from sklearn.datasets import make_blobs

    if not os.path.exists(path):
        os.mkdir(path)
    X, _ = make_blobs(n_samples=nb, n_features=d, centers=centers, random_state=1)

    with open(os.path.join(path, "data.f32bin"), "wb") as f:
        np.array([nb, d], dtype="uint32").tofile(f)
        X.astype("float32").tofile(f)


def create_manual_dataset_f32_only_data(data, path, dataset_name="data.f32bin"):
    """
    Creates a dataset from manually defined data and writes it to disk.

    Parameters
    ----------
    data: numpy.ndarray
        Manually defined data
    path: str
        Path to write the dataset to
    """
    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path, dataset_name), "wb") as f:
        np.array([data.shape[0], data.shape[1]], dtype="uint32").tofile(f)
        data.astype("float32").tofile(f)


def create_random_dataset_f32(nb, d, nq, k, path):
    """
    Creates a random float32 dataset containing both a dataset and queries against it, and then writes those to disk.

    Parameters
    ----------
    nb: int
        Number of points in the dataset
    d: int
        Dimension of the dataset
    nq: int
        Number of queries
    k: int
        Number of nearest neighbors to return
    path: str
        Path to write the dataset to
    """
    import sklearn.model_selection
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import NearestNeighbors

    # print(f"Preparing datasets with {nb} random points and {nq} queries.")
    if not os.path.exists(path):
        os.mkdir(path)
    X, _ = make_blobs(n_samples=nb + nq, n_features=d, centers=nq, random_state=1)

    data, queries = sklearn.model_selection.train_test_split(
        X, test_size=nq, random_state=1
    )

    with open(os.path.join(path, "data.f32bin"), "wb") as f:
        np.array([nb, d], dtype="uint32").tofile(f)
        data.astype("float32").tofile(f)
    with open(os.path.join(path, "queries"), "wb") as f:
        np.array([nq, d], dtype="uint32").tofile(f)
        queries.astype("float32").tofile(f)

    # print("Computing groundtruth")

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute").fit(
        data
    )
    D, I = nbrs.kneighbors(queries)
    with open(os.path.join(path, "gt"), "wb") as f:
        np.array([nq, k], dtype="uint32").tofile(f)
        I.astype("uint32").tofile(f)
        D.astype("float32").tofile(f)


def create_random_dataset_u8(nb, d, nq, k, path):
    """
    Creates a random uint8 dataset containing both a dataset and queries against it, and then writes those to disk.

    Parameters
    ----------
    nb: int
        Number of points in the dataset
    d: int
        Dimension of the dataset
    nq: int
        Number of queries
    k: int
        Number of nearest neighbors to return
    path: str
        Path to write the dataset to
    """
    import sklearn.model_selection
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import NearestNeighbors

    # print(f"Preparing datasets with {nb} random points and {nq} queries.")
    if not os.path.exists(path):
        os.mkdir(path)
    X, _ = make_blobs(n_samples=nb + nq, n_features=d, centers=nq, random_state=1)

    data, queries = sklearn.model_selection.train_test_split(
        X, test_size=nq, random_state=1
    )
    data = data.astype("uint8")
    queries = queries.astype("uint8")

    with open(os.path.join(path, "data.u8bin"), "wb") as f:
        np.array([nb, d], dtype="uint32").tofile(f)
        data.tofile(f)
    with open(os.path.join(path, "queries"), "wb") as f:
        np.array([nq, d], dtype="uint32").tofile(f)
        queries.tofile(f)

    # print("Computing groundtruth")

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute").fit(
        data
    )
    D, I = nbrs.kneighbors(queries)
    with open(os.path.join(path, "gt"), "wb") as f:
        np.array([nq, k], dtype="uint32").tofile(f)
        I.astype("uint32").tofile(f)
        D.astype("float32").tofile(f)
    return data


def create_schema(dimension_0_domain_max, dimension_1_domain_max):
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            *[
                tiledb.Dim(
                    name="__dim_0",
                    domain=(0, dimension_0_domain_max),
                    tile=max(1, min(3, dimension_0_domain_max)),
                    dtype="int32",
                ),
                tiledb.Dim(
                    name="__dim_1",
                    domain=(0, dimension_1_domain_max),
                    tile=max(1, min(3, dimension_1_domain_max)),
                    dtype="int32",
                ),
            ]
        ),
        attrs=[
            tiledb.Attr(name="values", dtype="float32", var=False, nullable=False),
        ],
        cell_order="col-major",
        tile_order="col-major",
        capacity=10000,
        sparse=False,
    )
    return schema


def create_array(path: str, data):
    schema = create_schema(
        data.shape[0] - 1,  # number of rows
        data.shape[1] - 1,  # number of cols
    )
    tiledb.Array.create(path, schema)
    with tiledb.open(path, "w") as A:
        A[:] = data


def accuracy(
    result, gt, external_ids_offset=0, updated_ids=None, only_updated_ids=False
):
    found = 0
    total = 0
    if updated_ids is not None:
        updated_ids_rev = {}
        for updated_id in updated_ids:
            updated_ids_rev[updated_ids[updated_id]] = updated_id
    for i in range(len(result)):
        if external_ids_offset != 0:
            temp_result = []
            for j in range(len(result[i])):
                temp_result.append(int(result[i][j] - external_ids_offset))
        elif updated_ids is not None:
            temp_result = []
            for j in range(len(result[i])):
                if result[i][j] in updated_ids:
                    raise ValueError(
                        f"Found updated id {result[i][j]} in query results."
                    )
                if only_updated_ids:
                    if result[i][j] not in updated_ids_rev:
                        raise ValueError(
                            f"Found not_updated_id {result[i][j]} in query results while expecting only_updated_ids."
                        )
                uid = updated_ids_rev.get(result[i][j])
                if uid is not None:
                    temp_result.append(int(uid))
                else:
                    temp_result.append(result[i][j])
        else:
            temp_result = result[i]
        total += len(temp_result)
        found += len(np.intersect1d(temp_result, gt[i]))
    return found / total


def check_equals(result_d, result_i, expected_result_d, expected_result_i):
    """
    Check that the results are equal to the expected results.

    Parameters
    ----------
    result_d: int
        The distances returned by the query
    result_i: int
        The indices returned by the query
    result_d_expected: int
        The expected distances
    result_i_expected: int
        The expected indices
    """
    assert np.array_equal(
        result_i, expected_result_i
    ), f"result_i: {result_i} != expected_result_i: {expected_result_i}"
    assert np.array_equal(
        result_d, expected_result_d
    ), f"result_d: {result_d} != expected_result_d: {expected_result_d}"


def random_name(name: str) -> str:
    """
    Generate random names for test array uris
    """
    suffix = "".join(random.choices(string.ascii_letters, k=10))
    return f"zzz_unittest_{name}_{suffix}"


def check_training_input_vectors(
    index_uri: str,
    expected_training_sample_size: int,
    expected_dimensions: int,
    config=None,
):
    training_input_vectors_uri = f"{index_uri}/{storage_formats[STORAGE_VERSION]['TRAINING_INPUT_VECTORS_ARRAY_NAME']}"
    with tiledb.open(training_input_vectors_uri, mode="r", config=config) as src_array:
        training_input_vectors = np.transpose(src_array[:, :]["values"])
        assert training_input_vectors.shape[0] == expected_training_sample_size
        assert training_input_vectors.shape[1] == expected_dimensions
        assert not np.isnan(training_input_vectors).any()


def move_local_index_to_new_location(index_uri):
    """
    Moves to the index to a new location on the computer. This helps test that there are no absolute
    paths in the index.
    """
    copied_index_uri = index_uri + "_copied"
    shutil.copytree(index_uri, copied_index_uri)
    shutil.rmtree(index_uri)
    return copied_index_uri


def quantize_embeddings_int8(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Quantizes embeddings to a lower precision.
    """
    ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))
    starts = ranges[0, :]
    steps = (ranges[1, :] - ranges[0, :]) / 255
    return ((embeddings - starts) / steps - 128).astype(np.int8)


def setUpCloudToken():
    token = os.getenv("TILEDB_REST_TOKEN")
    if os.getenv("TILEDB_CLOUD_HELPER_VAR"):
        token = os.getenv("TILEDB_CLOUD_HELPER_VAR")
    tiledb.cloud.login(token=token)


def create_cloud_uri(name):
    namespace, storage_path, _ = groups._default_ns_path_cred()
    storage_path = storage_path.replace("//", "/").replace("/", "//", 1)
    rand_name = random_name("vector_search")
    test_path = f"tiledb://{namespace}/{storage_path}/{rand_name}"
    return f"{test_path}/{name}"


def delete_uri(uri, config):
    with tiledb.scope_ctx(ctx_or_config=config):
        try:
            group = tiledb.Group(uri, "m")
        except tiledb.TileDBError as err:
            message = str(err)
            if "does not exist" in message:
                return
            else:
                raise err
        group.delete(recursive=True)


def load_metadata(index_uri):
    group = tiledb.Group(index_uri, "r")
    ingestion_timestamps = [
        int(x) for x in list(json.loads(group.meta.get("ingestion_timestamps", "[]")))
    ]
    base_sizes = [int(x) for x in list(json.loads(group.meta.get("base_sizes", "[]")))]
    group.close()
    return ingestion_timestamps, base_sizes
