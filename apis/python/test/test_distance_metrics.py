import json
import os

import numpy as np

from array_paths import *
from common import *

# import capsys
from common import load_metadata
import pytest
from sklearn.neighbors import NearestNeighbors


import tiledb.vector_search as vs
from tiledb.vector_search import Index
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

    print("Passed first test")

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

    captured = capfd.readouterr()
    print(captured.out)

    # index = index.consolidate_updates()
    # TODO(SC-46771): Investigate whether we should overwrite the existing metadata during the first
    # ingestion of Python indexes. I believe as it's currently written we have a bug here.
    # ingestion_timestamps, base_sizes = load_metadata(uri)
    # assert base_sizes == [5]
    # timestamp_5_minutes_from_now = int((time.time() + 5 * 60) * 1000)
    # timestamp_5_minutes_ago = int((time.time() - 5 * 60) * 1000)
    # assert ingestion_timestamps[0] > timestamp_5_minutes_ago and ingestion_timestamps[0] < timestamp_5_minutes_from_now

    vfs = tiledb.VFS()
    assert vfs.dir_size(uri) > 0
    Index.delete_index(uri=uri, config={})
    assert vfs.dir_size(uri) == 0


# def test_ivf_flat_cosine_vs_flat(tmp_path):
#     # Generate 1000 random vectors with 128 dimensions
#     num_vectors = 1000
#     vector_dim = 128
#     input_vectors = np.random.rand(num_vectors, vector_dim).astype(np.float32)

#     # Create IVF_FLAT index with cosine distance
#     ivf_flat_uri = os.path.join(tmp_path, "ivf_flat_cosine")
#     ivf_flat_index = ingest(
#         index_type="IVF_FLAT",
#         index_uri=ivf_flat_uri,
#         input_vectors=input_vectors,
#         distance_metric=vspy.DistanceMetric.COSINE,
#     )

#     # Create FLAT index
#     flat_uri = os.path.join(tmp_path, "flat")
#     flat_index_obj = ingest(
#         index_type="FLAT",
#         index_uri=flat_uri,
#         input_vectors=input_vectors,
#         distance_metric=vspy.DistanceMetric.COSINE
#     )

#     # Generate query vectors
#     num_queries = 10
#     k = 5
#     query_vectors = np.random.rand(num_queries, vector_dim).astype(np.float32)

#     # Query IVF_FLAT index
#     ivf_flat_distances, ivf_flat_ids = ivf_flat_index.query(query_vectors, k=k)

#     # Query FLAT index
#     flat_distances, flat_ids = flat_index_obj.query(query_vectors, k=k)

#     # Use sklearn for ground truth
#     nn_cosine = NearestNeighbors(n_neighbors=k, metric='cosine')
#     nn_cosine.fit(input_vectors)
#     sklearn_distances, sklearn_ids = nn_cosine.kneighbors(query_vectors)

#     # Compare results
#     np.testing.assert_allclose(ivf_flat_distances, sklearn_distances, atol=1e-4,
#                                err_msg="IVF_FLAT distances do not match sklearn")
#     np.testing.assert_allclose(ivf_flat_ids, sklearn_ids, atol=1,
#                                err_msg="IVF_FLAT ids do not match sklearn")

#     np.testing.assert_allclose(flat_distances, sklearn_distances, atol=1e-4,
#                                err_msg="FLAT distances do not match sklearn")
#     np.testing.assert_allclose(flat_ids, sklearn_ids, atol=1,
#                                err_msg="FLAT ids do not match sklearn")

#     # Compare IVF_FLAT and FLAT results
#     np.testing.assert_allclose(ivf_flat_distances, flat_distances, atol=1e-4,
#                                err_msg="IVF_FLAT distances do not match FLAT distances")
#     np.testing.assert_allclose(ivf_flat_ids, flat_ids, atol=1,
#                                err_msg="IVF_FLAT ids do not match FLAT ids")

#     print("All assertions passed. IVF_FLAT (Cosine) and FLAT index results match each other and sklearn's results.")

#     # Optionally, print some results for visual inspection
#     print("\nSample results for the first query:")
#     print("IVF_FLAT distances:", ivf_flat_distances[0])
#     print("FLAT distances:", flat_distances[0])
#     print("Sklearn distances:", sklearn_distances[0])
#     print("IVF_FLAT ids:", ivf_flat_ids[0])
#     print("FLAT ids:", flat_ids[0])
#     print("Sklearn ids:", sklearn_ids[0])

# def test_ivf_flat_cosine_vs_sklearn(tmp_path):
#     # Parameters
#     num_vectors = 10000
#     vector_dim = 128
#     k = 10
#     query_size = 100
#     partitions = 10  # number of clusters for IVF

#     # Generate random vectors
#     vectors = np.random.rand(num_vectors, vector_dim).astype(np.float32)

#     # Normalize vectors for cosine similarity
#     normalized_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

#     # Create IVF_FLAT index with Cosine distance
#     uri_cosine = os.path.join(tmp_path, "ivf_flat_cosine")
#     index_cosine = ingest(
#         index_type="IVF_FLAT",
#         index_uri=uri_cosine,
#         input_vectors=vectors,
#         distance_metric=vspy.DistanceMetric.COSINE,
#         partitions=partitions
#     )

#     # Create sklearn NearestNeighbors
#     nn_sklearn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
#     nn_sklearn.fit(normalized_vectors)

#     # Create a simple IVF implementation with sklearn
#     kmeans = KMeans(n_clusters=partitions, random_state=0).fit(normalized_vectors)
#     cluster_centers = kmeans.cluster_centers_
#     cluster_assignment = kmeans.labels_

#     # Generate query vectors
#     query_vectors = np.random.rand(query_size, vector_dim).astype(np.float32)
#     normalized_query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1)[:, np.newaxis]

#     # Query IVF_FLAT Cosine index
#     ivf_flat_distances, ivf_flat_ids = index_cosine.query(query_vectors, k=k)

#     # Query sklearn NearestNeighbors (exact search)
#     sklearn_distances, sklearn_ids = nn_sklearn.kneighbors(normalized_query_vectors)

#     # Simple IVF search with sklearn
#     def simple_ivf_search(query, k):
#         # Find nearest cluster
#         cluster_distances = np.linalg.norm(cluster_centers - query, axis=1)
#         nearest_cluster = np.argmin(cluster_distances)

#         # Get vectors in the nearest cluster
#         cluster_vectors = normalized_vectors[cluster_assignment == nearest_cluster]

#         # Perform exact search within the cluster
#         distances = np.dot(cluster_vectors, query)
#         indices = np.argsort(distances)[::-1][:k]
#         return distances[indices], indices

#     sklearn_ivf_distances = []
#     sklearn_ivf_ids = []
#     for query in normalized_query_vectors:
#         distances, indices = simple_ivf_search(query, k)
#         sklearn_ivf_distances.append(distances)
#         sklearn_ivf_ids.append(indices)

#     # Compute accuracy against exact search
#     accuracy_exact = np.mean([
#         len(set(ivf_flat_ids[i]).intersection(set(sklearn_ids[i]))) / k
#         for i in range(query_size)
#     ])

#     # Compute accuracy against simple IVF implementation
#     accuracy_ivf = np.mean([
#         len(set(ivf_flat_ids[i]).intersection(set(sklearn_ivf_ids[i]))) / k
#         for i in range(query_size)
#     ])

#     print(f"Accuracy against exact search: {accuracy_exact * 100:.2f}%")
#     print(f"Accuracy against simple IVF: {accuracy_ivf * 100:.2f}%")

#     # Assert accuracy is over a threshold (you might need to adjust this)
#     assert accuracy_exact > 0.8, f"Accuracy against exact search {accuracy_exact * 100:.2f}% is below 80%"
#     assert accuracy_ivf > 0.9, f"Accuracy against simple IVF {accuracy_ivf * 100:.2f}% is below 90%"

#     # Optionally, compare recall@k
#     def recall_at_k(true_ids, pred_ids, k):
#         return np.mean([len(set(true_ids[i][:k]).intersection(set(pred_ids[i][:k]))) / k for i in range(len(true_ids))])

#     recall = recall_at_k(sklearn_ids, ivf_flat_ids, k)
#     print(f"Recall@{k}: {recall * 100:.2f}%")


def test_ivf_flat_vs_flat_cosine(tmp_path):
    # Parameters
    num_vectors = 1000000
    vector_dim = 128
    k = 10
    query_size = 100
    partitions = 100  # for IVF_FLAT

    # Generate random vectors
    vectors = np.random.rand(num_vectors, vector_dim).astype(np.float32)

    # Normalize vectors for cosine similarity
    vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

    # Create IVF_FLAT index with Cosine distance
    uri_ivf_flat = os.path.join(tmp_path, "ivf_flat_cosine")
    index_ivf_flat = ingest(
        index_type="IVF_FLAT",
        index_uri=uri_ivf_flat,
        input_vectors=vectors,
        distance_metric=vspy.DistanceMetric.COSINE,
        partitions=partitions,
        training_sampling_policy=vs.ingestion.TrainingSamplingPolicy.RANDOM,
    )

    # Create FLAT index with Cosine distance
    uri_flat = os.path.join(tmp_path, "flat_cosine")
    index_flat = ingest(
        index_type="FLAT",
        index_uri=uri_flat,
        input_vectors=vectors,
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    # Generate query vectors
    query_vectors = np.random.rand(query_size, vector_dim).astype(np.float32)
    normalized_query_vectors = (
        query_vectors / np.linalg.norm(query_vectors, axis=1)[:, np.newaxis]
    )

    # Query FLAT Cosine index (ground truth)
    flat_distances, flat_ids = index_flat.query(query_vectors, k=k)

    # Compute recall@k for different nprobe values
    def recall_at_k(true_ids, pred_ids, k):
        return np.mean(
            [
                len(set(true_ids[i][:k]).intersection(set(pred_ids[i][:k]))) / k
                for i in range(len(true_ids))
            ]
        )

    nprobe_values = [1, 2, 4, 8, 16, 32, 64, partitions]  # Add more values if needed
    for nprobe in nprobe_values:
        ivf_flat_distances, ivf_flat_ids = index_ivf_flat.query(
            query_vectors, k=k, nprobe=nprobe
        )

        accuracy = np.mean(
            [
                len(set(ivf_flat_ids[i]).intersection(set(flat_ids[i]))) / k
                for i in range(query_size)
            ]
        )

        recall = recall_at_k(flat_ids, ivf_flat_ids, k)

        print(f"nprobe = {nprobe}:")
        print(f"  Accuracy of IVF_FLAT compared to FLAT: {accuracy * 100:.2f}%")
        print(f"  Recall@{k}: {recall * 100:.2f}%")

        # Optionally, compare distances
        relative_distance_error = np.mean(
            np.abs(ivf_flat_distances - flat_distances) / flat_distances
        )
        print(f"  Average relative distance error: {relative_distance_error:.4f}")

        # Compare query times
        import time

        start_time = time.time()
        index_ivf_flat.query(normalized_query_vectors, k=k, nprobe=nprobe)
        ivf_flat_time = time.time() - start_time

        print(f"  IVF_FLAT query time: {ivf_flat_time:.4f} seconds")
        print()

    # Query FLAT index for timing comparison
    start_time = time.time()
    index_flat.query(normalized_query_vectors, k=k)
    flat_time = time.time() - start_time

    print(f"FLAT query time: {flat_time:.4f} seconds")
    print(f"Max speed-up (nprobe=1): {flat_time / ivf_flat_time:.2f}x")

    # Assert accuracy is over a threshold for the highest nprobe value
    assert (
        accuracy > 0.8
    ), f"Accuracy {accuracy * 100:.2f}% is below 80% even with nprobe={partitions}"


def test_ivf_flat_vs_flat_l2(tmp_path):
    # Parameters
    num_vectors = 1000000
    vector_dim = 128
    k = 10
    query_size = 100
    partitions = 100  # for IVF_FLAT

    # Generate random vectors
    vectors = np.random.rand(num_vectors, vector_dim).astype(np.float32)

    # Create IVF_FLAT index with L2 distance
    uri_ivf_flat = os.path.join(tmp_path, "ivf_flat_l2")
    index_ivf_flat = ingest(
        index_type="IVF_FLAT",
        index_uri=uri_ivf_flat,
        input_vectors=vectors,
        distance_metric=vspy.DistanceMetric.L2,
        partitions=partitions,
    )

    # Create FLAT index with L2 distance
    uri_flat = os.path.join(tmp_path, "flat_l2")
    index_flat = ingest(
        index_type="FLAT",
        index_uri=uri_flat,
        input_vectors=vectors,
        distance_metric=vspy.DistanceMetric.L2,
    )

    # Generate query vectors
    query_vectors = np.random.rand(query_size, vector_dim).astype(np.float32)

    # Query FLAT L2 index (ground truth)
    flat_distances, flat_ids = index_flat.query(query_vectors, k=k)

    # Compute recall@k for different nprobe values
    def recall_at_k(true_ids, pred_ids, k):
        return np.mean(
            [
                len(set(true_ids[i][:k]).intersection(set(pred_ids[i][:k]))) / k
                for i in range(len(true_ids))
            ]
        )

    nprobe_values = [1, 2, 4, 8, 16, 32, 64, partitions]  # Add more values if needed
    for nprobe in nprobe_values:
        ivf_flat_distances, ivf_flat_ids = index_ivf_flat.query(
            query_vectors, k=k, nprobe=nprobe
        )

        accuracy = np.mean(
            [
                len(set(ivf_flat_ids[i]).intersection(set(flat_ids[i]))) / k
                for i in range(query_size)
            ]
        )

        recall = recall_at_k(flat_ids, ivf_flat_ids, k)

        print(f"nprobe = {nprobe}:")
        print(f"  Accuracy of IVF_FLAT compared to FLAT: {accuracy * 100:.2f}%")
        print(f"  Recall@{k}: {recall * 100:.2f}%")

        # Compare distances
        relative_distance_error = np.mean(
            np.abs(ivf_flat_distances - flat_distances) / flat_distances
        )
        print(f"  Average relative distance error: {relative_distance_error:.4f}")

        # Compare query times
        import time

        start_time = time.time()
        index_ivf_flat.query(query_vectors, k=k, nprobe=nprobe)
        ivf_flat_time = time.time() - start_time

        print(f"  IVF_FLAT query time: {ivf_flat_time:.4f} seconds")
        print()

    # Query FLAT index for timing comparison
    start_time = time.time()
    index_flat.query(query_vectors, k=k)
    flat_time = time.time() - start_time

    print(f"FLAT query time: {flat_time:.4f} seconds")
    print(f"Max speed-up (nprobe=1): {flat_time / ivf_flat_time:.2f}x")

    # Assert accuracy is over a threshold for the highest nprobe value
    assert (
        accuracy > 0.8
    ), f"Accuracy {accuracy * 100:.2f}% is below 80% even with nprobe={partitions}"


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
    index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
    index = ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.L2,
    )


def test_vamana_create_cosine(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")
    with pytest.raises(RuntimeError):
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

# def test_ivfpq_create_l2(tmp_path):
#     index_uri = os.path.join(tmp_path, "sift10k_flat_L2")
#     index = ingest(
#         index_type="IVFPQ",
#         index_uri=index_uri,
#         source_uri=siftsmall_uri,
#         source_type="FVEC",
#         distance_metric=vspy.DistanceMetric.L2,
#     )

# def test_ivfpq_create_cosine(tmp_path):
#     index_uri = os.path.join(tmp_path, "sift10k_flat_COSINE")
#     with pytest.raises(RuntimeError):
#         ingest(
#             index_type="IVFPQ",
#             index_uri=index_uri,
#             source_uri=siftsmall_uri,
#             source_type="FVEC",
#             distance_metric=vspy.DistanceMetric.COSINE,
#         )
