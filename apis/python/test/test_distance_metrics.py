import os

import numpy as np
import pytest
from array_paths import *
from common import *
from sklearn.neighbors import NearestNeighbors

from tiledb.cloud.dag import Mode
from tiledb.vector_search import Index
from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search import ivf_flat_index
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.utils import MAX_UINT64
from tiledb.vector_search.utils import load_fvecs

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

    minimum_accuracy = MINIMUM_ACCURACY
    index_uri = os.path.join(tmp_path, "array_IVF_FLAT")
    index = ingest(
        index_type="IVF_FLAT",
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
    index_ram = IVFFlatIndex(uri=index_uri)
    _, result = index_ram.query(queries, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > minimum_accuracy

    index_ram = IVFFlatIndex(uri=index_uri, memory_budget=int(size / 10))
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
        partitions=1,
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
    distances, ids = index.query(
        queries=np.array([[2, 2, 2, 2]], dtype=np.float32), k=5
    )
    assert np.array_equal(ids, np.array([[4, 2, 3, 1, 0]], dtype=np.uint64))
    sorted_distances = np.sort(distances)
    assert np.allclose(distances, sorted_distances, 1e-4)


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

    distances, ids = index.query(
        queries=np.array([[2, 2, 2, 2]], dtype=np.float32), k=5
    )
    assert np.array_equal(ids, np.array([[4, 2, 3, 1, 0]], dtype=np.uint64))
    sorted_distances = np.sort(distances)
    assert np.allclose(distances, sorted_distances, 1e-4)

    # Update the index with a new vector
    index.update(
        vector=np.array([2, 2, 2, 2], dtype=vector_type),
        external_id=5,
    )
    expected_distances = np.append(expected_distances, 0.0)

    # delete the vector with external id 3
    index.delete(external_id=3)

    # consolidate the updates
    index = index.consolidate_updates()

    # Query the index
    query_and_check(
        index,
        np.array([[2, 2, 2, 2]], dtype=np.float32),
        3,
        {2, 4, 5},
        expected_distances=expected_distances,
        nprobe=partitions,
    )

    distances, ids = index.query(
        queries=np.array([[2, 2, 2, 2]], dtype=np.float32), k=5
    )
    assert np.array_equal(ids, np.array([[5, 4, 2, 1, 0]], dtype=np.uint64))
    sorted_distances = np.sort(distances)
    assert np.allclose(distances, sorted_distances, 1e-4)


def test_ivf_flat_cosine_simple_normalized(tmp_path):
    # Create 5 input vectors and normalize them
    input_vectors = np.array(
        [
            normalize_vector([1, 0, 0, 0]),
            normalize_vector([1, 1, 0, 0]),
            normalize_vector([32, 41, 30, 0]),
            normalize_vector([1, 5, 3, 0]),
            normalize_vector([4, 4, 4, 0]),
        ],
        dtype=np.float32,
    )

    index_uri = os.path.join(tmp_path, "ivf_flat_cosine")
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=input_vectors,
        distance_metric=vspy.DistanceMetric.COSINE,
        partitions=1,
        normalized=True,
    )

    query_vector = normalize_vector(np.array([2, 2, 2, 2], dtype=np.float32))

    expected_distances = np.array([0.500000, 0.292893, 0.142262, 0.239361, 0.133975])

    query_and_check(
        index,
        query_vector.reshape(1, -1),
        3,
        {2, 3, 4},
        expected_distances=expected_distances,
        nprobe=2,
    )

    distances, ids = index.query(queries=query_vector.reshape(1, -1), k=5)
    assert np.array_equal(ids, np.array([[4, 2, 3, 1, 0]], dtype=np.uint64))
    sorted_distances = np.sort(distances)
    assert np.allclose(distances, sorted_distances, atol=1e-4)


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
    distances = np.sqrt(distances)

    assert np.allclose(
        distances_sum_of_squares, distances, 1e-4
    ), "Sum of squares distances do not match"
    assert np.array_equal(ids_sum_of_squares, ids), "Sum of squares ids do not match"


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
