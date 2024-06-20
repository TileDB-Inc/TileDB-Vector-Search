import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import tiledb.vector_search as vs
import os
import shutil
from tiledb.vector_search.utils import load_fvecs
from array_paths import *
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search import _tiledbvspy as vspy

siftsmall_uri = siftsmall_inputs_file
queries_uri = siftsmall_query_file

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def test_cosine_similarity(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_FLAT")
    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.COSINE
    )

    dataset_vectors = load_fvecs(siftsmall_uri)
    query_vectors = load_fvecs(queries_uri)    

    
    # sklearn
    nn_cosine_sklearn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_cosine_sklearn.fit(dataset_vectors)
    distances_sklearn, indices_sklearn = nn_cosine_sklearn.kneighbors(query_vectors)
    
    # Your library
    indices_yourlib = index.query(query_vectors, k=5)
    # Compare results
    assert np.allclose(distances_sklearn, indices_yourlib[0], 0.01), "Cosine similarity distances do not match"
    assert np.array_equal(indices_sklearn, indices_yourlib[1]), "Cosine similarity indices do not match"

def test_inner_product(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_IP")
    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=siftsmall_uri,
        source_type="FVEC",
        distance_metric=vspy.DistanceMetric.INNER_PRODUCT
    )

    dataset_vectors = load_fvecs(siftsmall_uri)
    query_vectors = load_fvecs(queries_uri)
    
    # sklearn calculation, assuming IP means we are comparing direct dot products
    inner_products_sklearn = np.dot(query_vectors, dataset_vectors.T)
    
    # Sorting inner products from sklearn in ascending order for each query
    sorted_inner_products_sklearn = np.sort(inner_products_sklearn, axis=1)
    
    # Querying your custom index, which is expected to return distances and indices
    results = index.query(query_vectors, k=5)
    inner_products_yourlib = results[0]  # Assuming results are [distances, indices]
    
    # Printing results for verification
    # print("Custom Library Inner Products:\n", inner_products_yourlib)
    # print("Sklearn Sorted Inner Products:\n", sorted_inner_products_sklearn)
    
    # Assuming that your custom library returns sorted distances in descending order, you might need to sort them as well
    sorted_inner_products_yourlib = np.sort(inner_products_yourlib, axis=1)
    
    # Compare results
    # go through each query and compare the sorted inner products
    for i in range(len(sorted_inner_products_sklearn)):
        # only keep lowest 5 inner products in each query
        compare = sorted_inner_products_sklearn[i][:5]
        assert np.allclose(compare, sorted_inner_products_yourlib[i], 0.01), "Inner products do not match"

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
    # sklearn
    nn_l2 = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_l2.fit(dataset_vectors)
    distances_l2, indices_l2 = nn_l2.kneighbors(query_vectors)
    
    distances, indices = index.query(query_vectors, k=5)
    # apply sqrt to distances
    distances = np.sqrt(distances)
    assert np.allclose(distances_l2, distances, 0.01), "L2 distances do not match"
    # Compare results
    assert np.array_equal(indices_l2, indices), "L2 indices do not match"

def test_wrong_distance_metric(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_flat_IDK")
    with pytest.raises(AttributeError):
        index = ingest(
            index_type="FLAT",
            index_uri=index_uri,
            source_uri=siftsmall_uri,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.IDK
        )
    
def test_wrong_type_with_distance_metric(tmp_path):
    index_uri = os.path.join(tmp_path, "sift10k_IVF_FLAT_COSINE")
    with pytest.raises(ValueError):
        index = ingest(
            index_type="IVF_FLAT",
            index_uri=index_uri,
            source_uri=siftsmall_uri,
            source_type="FVEC",
            distance_metric=vspy.DistanceMetric.COSINE
        )