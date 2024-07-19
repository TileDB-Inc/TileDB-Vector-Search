import os

import numpy as np
import pytest
from array_paths import *
from sklearn.neighbors import NearestNeighbors

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
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
    )

    dataset_vectors = load_fvecs(siftsmall_inputs_file)
    query_vectors = load_fvecs(siftsmall_query_file)

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
            distance_metric=vspy.DistanceMetric.COSINE,
        )


def test_vamana_create_l2(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
    ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.L2,
    )


def test_vamana_create_cosine(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")
    with pytest.raises(ValueError):
        ingest(
            index_type="VAMANA",
            index_uri=index_uri,
            source_uri=siftsmall_inputs_file,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.COSINE,
        )


def test_ivfpq_create_l2(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
    ingest(
        index_type="IVF_PQ",
        index_uri=index_uri,
        source_uri=siftsmall_inputs_file,
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
            source_uri=siftsmall_inputs_file,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.COSINE,
            num_subspaces=2,
        )
