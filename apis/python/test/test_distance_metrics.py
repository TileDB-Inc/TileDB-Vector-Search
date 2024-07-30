import json
import os

import numpy as np
import pytest
from array_paths import *
from common import *

# import capsys
from common import load_metadata
from sklearn.neighbors import NearestNeighbors

from tiledb.cloud.dag import Mode
from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search import ivf_flat_index
from tiledb.vector_search.index import DATASET_TYPE
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.utils import MAX_UINT64
from tiledb.vector_search.utils import is_type_erased_index
from tiledb.vector_search.utils import load_fvecs

siftsmall_uri = siftsmall_inputs_file
queries_uri = siftsmall_query_file


def check_default_metadata(
    uri, expected_vector_type, expected_storage_version, expected_index_type
):
    group = tiledb.Group(uri, "r", ctx=tiledb.Ctx(None))
    assert "dataset_type" in group.meta
    assert group.meta["dataset_type"] == DATASET_TYPE
    assert type(group.meta["dataset_type"]) == str

    assert "dtype" in group.meta
    assert group.meta["dtype"] == np.dtype(expected_vector_type).name
    assert type(group.meta["dtype"]) == str

    assert "storage_version" in group.meta
    assert group.meta["storage_version"] == expected_storage_version
    assert type(group.meta["storage_version"]) == str

    assert "index_type" in group.meta
    assert group.meta["index_type"] == expected_index_type
    assert type(group.meta["index_type"]) == str

    assert "base_sizes" in group.meta
    assert group.meta["base_sizes"] == json.dumps([0])
    assert type(group.meta["base_sizes"]) == str

    assert "ingestion_timestamps" in group.meta
    assert group.meta["ingestion_timestamps"] == json.dumps([0])
    assert type(group.meta["ingestion_timestamps"]) == str

    if not is_type_erased_index(expected_index_type):
        # NOTE(paris): Type-erased indexes do not write has_updates.
        assert "has_updates" in group.meta
        assert group.meta["has_updates"] == 0
        assert type(group.meta["has_updates"]) == np.int64
    else:
        assert "has_updates" not in group.meta


def query_and_check_distances(
    index, queries, k, expected_distances, expected_ids, **kwargs
):
    for _ in range(1):
        distances, ids = index.query(queries, k=k, **kwargs)
        assert np.array_equal(ids, expected_ids)
        assert np.array_equal(distances, expected_distances)


def query_and_check(index, queries, k, expected, expected_distances=None, **kwargs):
    for _ in range(3):
        result_d, result_i = index.query(queries, k=k, **kwargs)

        # Check if the expected IDs are a subset of the result
        assert expected.issubset(
            set(result_i[0])
        ), f"Expected IDs {expected} are not a subset of result IDs {set(result_i[0])}"

        # If expected_distances is provided, check the distances
        if expected_distances is not None:
            expected_dict = dict(
                zip(range(len(expected_distances)), expected_distances)
            )

            result_dict = dict(zip(result_i[0], result_d[0]))

            for id in expected.intersection(set(result_i[0])):
                np.testing.assert_allclose(
                    result_dict[id],
                    expected_dict[id],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Distance mismatch for ID {id}",
                )


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


MINIMUM_ACCURACY = 0.85
MINIMUM_ACCURACY_IVF_PQ = 0.75


def test_ivf_flat_ingestion_cosine(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    k = 10
    size = 10000
    dimensions = 127
    partitions = 100
    nqueries = 100
    nprobe = 20
    num_subspaces = 127

    create_random_dataset_f32(
        nb=size,
        d=dimensions,
        nq=nqueries,
        k=k,
        path=dataset_dir,
        distance_metric="cosine",
    )
    dtype = np.float32

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index_type = "IVF_FLAT"
    index_class = IVFFlatIndex
    minimum_accuracy = (
        MINIMUM_ACCURACY_IVF_PQ if index_type == "IVF_PQ" else MINIMUM_ACCURACY
    )
    index_uri = os.path.join(tmp_path, f"array_{index_type}")
    index = ingest(
        index_type=index_type,
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.f32bin"),
        partitions=partitions,
        input_vectors_per_work_item=int(size / 10),
        num_subspaces=num_subspaces,
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    _, result = index.query(queries, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > minimum_accuracy

    index_uri = move_local_index_to_new_location(index_uri)
    index_ram = index_class(uri=index_uri)
    _, result = index_ram.query(queries, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > minimum_accuracy

    index_ram = index_class(uri=index_uri, memory_budget=int(size / 10))
    _, result = index_ram.query(queries, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > minimum_accuracy

    _, result = index_ram.query(
        queries,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > minimum_accuracy

    _, result = index_ram.query(queries, k=k, nprobe=nprobe, mode=Mode.LOCAL)
    assert accuracy(result, gt_i) > minimum_accuracy


def test_ivf_flat_cosine_simple(tmp_path):
    # Create 5 input vectors

    input_vectors = np.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [32, 41, 30, 0], [1, 5, 3, 0], [4, 4, 4, 0]],
        dtype=np.float32,
    )

    # Create IVF_FLAT index with cosine distance
    index_uri = os.path.join(tmp_path, "ivf_flat_cosine")
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=input_vectors,
        distance_metric=vspy.DistanceMetric.COSINE,
        partitions=1,  # Using 2 partitions for this small dataset
    )

    # Create a query vector
    np.array([[2, 2, 2, 2]], dtype=np.float32)

    expected_distances = np.array([0.500000, 0.292893, 0.142262, 0.239361, 0.133975])

    query_and_check(
        index,
        np.array([[2, 2, 2, 2]], dtype=np.float32),
        3,
        {2, 3, 4},
        expected_distances=expected_distances,
        nprobe=2,
    )


def test_vamana_cosine_simple(tmp_path):
    # Create 5 input vectors

    input_vectors = np.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [32, 41, 30, 0], [1, 5, 3, 0], [4, 4, 4, 0]],
        dtype=np.float32,
    )

    # Create IVF_FLAT index with cosine distance
    index_uri = os.path.join(tmp_path, "vamana_cosine_simple")
    index = ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        input_vectors=input_vectors,
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    # Create a query vector
    np.array([[2, 2, 2, 2]], dtype=np.float32)

    expected_distances = np.array([0.500000, 0.292893, 0.142262, 0.239361, 0.133975])

    query_and_check(
        index,
        np.array([[2, 2, 2, 2]], dtype=np.float32),
        3,
        {2, 3, 4},
        expected_distances=expected_distances,
        nprobe=2,
    )


def test_ivf_flat_index(capfd, tmp_path):
    partitions = 10
    uri = os.path.join(tmp_path, "array")
    vector_type = np.dtype(np.float32)
    index = ivf_flat_index.create(
        uri=uri,
        dimensions=4,
        vector_type=vector_type,
        partitions=partitions,
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    ingestion_timestamps, base_sizes = load_metadata(uri)
    assert base_sizes == [0]
    assert ingestion_timestamps == [0]

    query_and_check(
        index,
        np.array([[2, 2, 2, 2]], dtype=np.float32),
        3,
        {MAX_UINT64},
        nprobe=partitions,
    )
    # check_default_metadata(uri, vector_type, STORAGE_VERSION, "IVF_FLAT")

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([1, 0, 0, 0], dtype=vector_type)
    update_vectors[1] = np.array([1, 1, 0, 0], dtype=vector_type)
    update_vectors[2] = np.array([32, 41, 30, 0], dtype=vector_type)
    update_vectors[3] = np.array([1, 5, 3, 0], dtype=vector_type)
    update_vectors[4] = np.array([4, 4, 4, 0], dtype=vector_type)
    index.update_batch(vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))

    expected_distances = np.array([0.500000, 0.292893, 0.142262, 0.239361, 0.133975])

    query_and_check(
        index,
        np.array([[2, 2, 2, 2]], dtype=np.float32),
        3,
        {2, 3, 4},
        expected_distances=expected_distances,
        nprobe=partitions,
    )

    index = index.consolidate_updates()

    query_and_check(
        index,
        np.array([[2, 2, 2, 2]], dtype=np.float32),
        3,
        {2, 3, 4},
        expected_distances=expected_distances,
        nprobe=partitions,
    )


def test_cosine_DISTANCE(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_FLAT")
    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    dataset_vectors = load_fvecs(siftsmall_uri)
    query_vectors = load_fvecs(queries_uri)

    nn_cosine_sklearn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn_cosine_sklearn.fit(dataset_vectors)
    distances_sklearn, ids_sklearn = nn_cosine_sklearn.kneighbors(query_vectors)

    distances, ids = index.query(query_vectors, k=5)

    assert np.allclose(
        distances_sklearn, distances, 1e-4
    ), "Cosine distances do not match"
    assert np.array_equal(ids_sklearn, ids), "Cosine distance ids do not match"


def test_inner_product_distances(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_IP")
    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.INNER_PRODUCT,
    )

    dataset_vectors = load_fvecs(siftsmall_uri)
    query_vectors = load_fvecs(queries_uri)

    inner_products_sklearn = np.dot(query_vectors, dataset_vectors.T)

    sorted_inner_products_sklearn = np.sort(inner_products_sklearn, axis=1)[:, ::-1]

    distances, _ = index.query(query_vectors, k=5)

    # multiply distances by -1 to get inner products, since library returns negative inner products
    distances = -1 * distances

    for i in range(len(sorted_inner_products_sklearn)):
        compare = sorted_inner_products_sklearn[i][:5]
        assert np.allclose(
            compare, distances[i], 1e-4
        ), "Inner product distances do not match"


def test_l2_distance(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
    )

    dataset_vectors = load_fvecs(siftsmall_uri)
    query_vectors = load_fvecs(queries_uri)

    nn_l2 = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn_l2.fit(dataset_vectors)
    distances_l2, ids_l2 = nn_l2.kneighbors(query_vectors)

    distances, ids = index.query(query_vectors, k=5)
    distances = np.sqrt(distances)
    assert np.allclose(distances_l2, distances, 1e-4), "L2 distances do not match"
    assert np.array_equal(ids_l2, ids), "L2 ids do not match"


def test_wrong_distance_metric(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_IDK")
    with pytest.raises(AttributeError):
        ingest(
            index_type="FLAT",
            index_uri=index_uri,
            source_uri=siftsmall_uri,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.IDK,
        )


def test_wrong_type_with_distance_metric(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_IVF_FLAT_COSINE")
    with pytest.raises(ValueError):
        ingest(
            index_type="IVF_FLAT",
            index_uri=index_uri,
            source_uri=siftsmall_uri,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.INNER_PRODUCT,
        )


def test_vamana_create_l2(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_L22")
    ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.L2,
    )


def test_vamana_create_cosine(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")
    ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.COSINE,
    )


def test_ivf_flat_create_cosine_numpy(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")

    # create index with numpy input_vectors
    input_vectors = load_fvecs(siftsmall_uri)

    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=input_vectors,
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    # query
    query_vectors = load_fvecs(queries_uri)
    distances, ids = index.query(query_vectors, k=5)

    # now create index with L2 and print results

    index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
    index2 = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=input_vectors,
    )

    distances2, ids2 = index2.query(query_vectors, k=5)


def test_ivfpq_create_l2(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
    ingest(
        index_type="IVF_PQ",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.L2,
        num_subspaces=2,
    )


def test_ivfpq_create_cosine(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")
    with pytest.raises(ValueError):
        ingest(
            index_type="IVF_PQ",
            index_uri=index_uri,
            source_uri=siftsmall_uri,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.COSINE,
            num_subspaces=2,
        )
