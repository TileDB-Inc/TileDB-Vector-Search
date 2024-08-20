import os

import numpy as np
import pytest
from array_paths import *
from common import *
from sklearn.neighbors import NearestNeighbors

from tiledb.vector_search import Index
from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.utils import load_fvecs


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def test_cosine_distance(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_FLAT")
    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    dataset_vectors = load_fvecs(siftsmall_inputs_file)
    query_vectors = load_fvecs(siftsmall_query_file)

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
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.INNER_PRODUCT,
    )

    dataset_vectors = load_fvecs(siftsmall_inputs_file)
    query_vectors = load_fvecs(siftsmall_query_file)

    inner_products_sklearn = np.dot(query_vectors, dataset_vectors.T)

    sorted_inner_products_sklearn = np.sort(inner_products_sklearn, axis=1)[:, ::-1]

    distances, _ = index.query(query_vectors, k=5)

    for i in range(len(sorted_inner_products_sklearn)):
        compare = sorted_inner_products_sklearn[i][:5]
        assert np.allclose(
            compare, distances[i], 1e-4
        ), "Inner product distances do not match"


def test_sum_of_squares_distance(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_sum_of_squares")
    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
    )

    dataset_vectors = load_fvecs(siftsmall_inputs_file)
    query_vectors = load_fvecs(siftsmall_query_file)

    nn_sum_of_squares = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn_sum_of_squares.fit(dataset_vectors)
    distances_sum_of_squares, ids_sum_of_squares = nn_sum_of_squares.kneighbors(
        query_vectors
    )

    distances, ids = index.query(query_vectors, k=5)
    assert np.allclose(
        distances_sum_of_squares, distances, 1e-4
    ), "L2 distances do not match"
    assert np.array_equal(ids_sum_of_squares, ids), "L2 ids do not match"


def test_wrong_distance_metric(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_IDK")
    with pytest.raises(AttributeError):
        ingest(
            index_type="FLAT",
            index_uri=index_uri,
            source_uri=siftsmall_inputs_file,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.IDK,
        )


def test_wrong_type_with_distance_metric(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_IVF_FLAT_COSINE")
    with pytest.raises(ValueError):
        ingest(
            index_type="IVF_FLAT",
            index_uri=index_uri,
            source_uri=siftsmall_inputs_file,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.INNER_PRODUCT,
        )


def test_vamana_create_sum_of_squares(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_sum_of_squares2")
    ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.SUM_OF_SQUARES,
    )


def test_vamana_create_cosine(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")
    ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.COSINE,
    )


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def test_ivf_flat_create_cosine_numpy(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")

    # Load input vectors
    input_vectors = load_fvecs(siftsmall_inputs_file)

    # Create index with numpy input_vectors
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=input_vectors,
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    # Load query vectors
    query_vectors = load_fvecs(siftsmall_query_file)

    # Query
    k = 5
    distances, ids = index.query(query_vectors, k=k)

    # Compute cosine distances manually and compare
    for i, query in enumerate(query_vectors):
        for j, idx in enumerate(ids[i]):
            manual_distance = cosine_distance(query, input_vectors[idx])
            np.testing.assert_allclose(
                manual_distance,
                distances[i][j],
                rtol=1e-4,
                err_msg=f"Mismatch for query {i}, neighbor {j}",
            )

    # Verify that distances are sorted
    assert np.all(
        np.diff(distances, axis=1) >= 0
    ), "Distances are not sorted in ascending order"

    # Create index with L2 distance for comparison
    index_uri_l2 = os.path.join(tmp_path, "sift10k_flat_L2")
    index_l2 = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri_l2,
        input_vectors=input_vectors,
        distance_metric=vspy.DistanceMetric.L2,
    )

    distances_l2, ids_l2 = index_l2.query(query_vectors, k=k)

    # Verify that L2 results are different from cosine results
    assert not np.array_equal(
        ids, ids_l2
    ), "Cosine and L2 queries returned the same indices"

    for i, query in enumerate(query_vectors):
        for j, idx in enumerate(ids_l2[i]):
            manual_distance = l2_distance(query, input_vectors[idx])
            np.testing.assert_allclose(
                manual_distance,
                distances_l2[i][j],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"L2 distance mismatch for query {i}, neighbor {j}",
            )

    Index.delete_index(uri=index_uri, config={})
    Index.delete_index(uri=index_uri_l2, config={})


def test_vamana_create_inner_product(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
    with pytest.raises(ValueError):
        ingest(
            index_type="VAMAAN",
            index_uri=index_uri,
            source_uri=siftsmall_inputs_file,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.INNER_PRODUCT,
        )


def sum_of_squares_distance(a, b):
    return np.sum((a - b) ** 2)


def test_ivfpq_create_sum_of_squares(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_sum_of_squares")
    ingest(
        index_type="IVF_PQ",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.SUM_OF_SQUARES,
        num_subspaces=2,
    )


def test_ivfpq_create_cosine(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")
    with pytest.raises(ValueError):
        ingest(
            index_type="IVF_PQ",
            index_uri=index_uri,
            source_uri=siftsmall_inputs_file,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.COSINE,
            num_subspaces=2,
        )


def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def test_flat_l2_distance(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.L2,
    )

    dataset_vectors = load_fvecs(siftsmall_inputs_file)
    query_vectors = load_fvecs(siftsmall_query_file)

    nn_l2 = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn_l2.fit(dataset_vectors)
    distances_l2_sklearn, ids_l2_sklearn = nn_l2.kneighbors(query_vectors)

    distances, ids = index.query(query_vectors, k=5)

    assert np.allclose(
        distances_l2_sklearn, distances, rtol=1e-5, atol=1e-8
    ), "L2 distances do not match"
    assert np.array_equal(ids_l2_sklearn, ids), "L2 ids do not match"


def test_ivf_flat_l2_distance(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_ivf_flat_L2")
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.L2,
        partitions=10,
    )

    dataset_vectors = load_fvecs(siftsmall_inputs_file)
    query_vectors = load_fvecs(siftsmall_query_file)

    k = 5
    distances, ids = index.query(query_vectors, k=k, nprobe=10)

    for i, query in enumerate(query_vectors):
        for j, idx in enumerate(ids[i]):
            manual_distance = l2_distance(query, dataset_vectors[idx])
            np.testing.assert_allclose(
                manual_distance,
                distances[i][j],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"L2 distance mismatch for query {i}, neighbor {j}",
            )

    assert np.all(
        np.diff(distances, axis=1) >= 0
    ), "Distances are not sorted in ascending order"


def test_vamana_l2_distance(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_vamana_L2")
    index = ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.L2,
    )

    dataset_vectors = load_fvecs(siftsmall_inputs_file)
    query_vectors = load_fvecs(siftsmall_query_file)

    k = 5
    distances, ids = index.query(query_vectors, k=k)

    for i, query in enumerate(query_vectors):
        for j, idx in enumerate(ids[i]):
            manual_distance = l2_distance(query, dataset_vectors[idx])
            np.testing.assert_allclose(
                manual_distance,
                distances[i][j],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"L2 distance mismatch for query {i}, neighbor {j}",
            )

    assert np.all(
        np.diff(distances, axis=1) >= 0
    ), "Distances are not sorted in ascending order"
