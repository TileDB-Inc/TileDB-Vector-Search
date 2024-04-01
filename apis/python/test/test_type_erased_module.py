import logging

import numpy as np
from array_paths import *

from tiledb.vector_search import _tiledbvspy as vspy

ctx = vspy.Ctx({})

ctx = vspy.Ctx({})

def test_inplace_build_query_IndexVamana():
    opt_l = 100
    k_nn = 10

    a = vspy.IndexVamana(id_type="uint32", adjacency_row_index_type="uint32", feature_type="float32")

    training_set = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert training_set.feature_type_string() == "float32"
    query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    assert query_set.feature_type_string() == "float32"
    groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert groundtruth_set.feature_type_string() == "uint64"

    a.train(training_set)
    s, t = a.query(query_set, k_nn, opt_l)

    intersections = vspy.count_intersections(t, groundtruth_set, k_nn)

    nt = np.double(t.num_vectors()) * np.double(k_nn)
    recall = intersections / nt

    assert recall == 1.0

def test_construct_IndexVamana():
    a = vspy.IndexVamana()
    assert a.feature_type_string() == "any"
    assert a.id_type_string() == "uint32"
    assert a.adjacency_row_index_type_string() == "uint32"
    assert a.dimension() == 0

    a = vspy.IndexVamana(feature_type="float32")
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "uint32"
    assert a.adjacency_row_index_type_string() == "uint32"
    assert a.dimension() == 0

    a = vspy.IndexVamana(feature_type="uint8", id_type="uint64", adjacency_row_index_type="int64")
    assert a.feature_type_string() == "uint8"
    assert a.id_type_string() == "uint64"
    assert a.adjacency_row_index_type_string() == "int64"
    assert a.dimension() == 0

    a = vspy.IndexVamana(feature_type="float32", id_type="int64", adjacency_row_index_type="uint64")
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "int64"
    assert a.adjacency_row_index_type_string() == "uint64"
    assert a.dimension() == 0

def test_construct_IndexVamana_with_empty_vector(tmp_path):
    opt_l = 100
    k_nn = 10
    index_uri = os.path.join(tmp_path, "array")
    dimensions = 128
    feature_type = "float32"
    id_type = "uint64"
    adjacency_row_index_type = "uint64"

    # First create an empty index.
    a = vspy.IndexVamana(feature_type=feature_type, id_type=id_type, adjacency_row_index_type=adjacency_row_index_type, dimension=dimensions)
    empty_vector = vspy.FeatureVectorArray(dimensions, 0, feature_type, id_type)
    a.train(empty_vector)
    a.write_index(ctx, index_uri)

    # Then load it again, retrain, and query.
    a = vspy.IndexVamana(ctx, index_uri)
    training_set = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert training_set.feature_type_string() == "float32"
    query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    assert query_set.feature_type_string() == "float32"
    groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert groundtruth_set.feature_type_string() == "uint64"

    a.train(training_set)
    
    s, t = a.query(query_set, k_nn, opt_l)
    intersections = vspy.count_intersections(t, groundtruth_set, k_nn)
    nt = np.double(t.num_vectors()) * np.double(k_nn)
    recall = intersections / nt
    assert recall == 1.0

def test_inplace_build_query_IndexVamana():
    opt_l = 100
    k_nn = 10

    a = vspy.IndexVamana(id_type="uint32", adjacency_row_index_type="uint32", feature_type="float32")

    training_set = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert training_set.feature_type_string() == "float32"
    query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    assert query_set.feature_type_string() == "float32"
    groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert groundtruth_set.feature_type_string() == "uint64"

    a.train(training_set)
    s, t = a.query(query_set, k_nn, opt_l)
    intersections = vspy.count_intersections(t, groundtruth_set, k_nn)

    nt = np.double(t.num_vectors()) * np.double(k_nn)
    recall = intersections / nt

    assert recall == 1.0

def test_construct_IndexIVFFlat():
    a = vspy.IndexIVFFlat()
    assert a.feature_type_string() == "any"
    assert a.id_type_string() == "uint32"
    assert a.px_type_string() == "uint32"

    a = vspy.IndexIVFFlat(feature_type="float32")
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "uint32"
    assert a.px_type_string() == "uint32"

    a = vspy.IndexIVFFlat(feature_type="uint8", id_type="uint64", px_type="int64")
    assert a.feature_type_string() == "uint8"
    assert a.id_type_string() == "uint64"
    assert a.px_type_string() == "int64"

    a = vspy.IndexIVFFlat(feature_type="float32", id_type="int64", px_type="uint64")
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "int64"
    assert a.px_type_string() == "uint64"


def test_inplace_build_infinite_query_IndexIVFFlat():
    k_nn = 10
    nprobe = 32

    for nprobe in [8, 32]:
        a = vspy.IndexIVFFlat(id_type="uint32", px_type="uint32")

        training_set = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
        assert training_set.feature_type_string() == "float32"

        query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
        assert query_set.feature_type_string() == "float32"

        groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
        assert groundtruth_set.feature_type_string() == "uint64"

        a.train(training_set, "random")
        a.add(training_set)
        s, t = a.query_infinite_ram(query_set, k_nn, nprobe)

        intersections = vspy.count_intersections(t, groundtruth_set, k_nn)

        nt = np.double(t.num_vectors()) * np.double(k_nn)
        recall = intersections / nt

        logging.info(f"nprobe = {nprobe}, recall={recall}")

        if nprobe == 8:
            assert recall > 0.925
        if nprobe == 32:
            assert recall >= 0.999
