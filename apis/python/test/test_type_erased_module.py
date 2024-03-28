import logging

import numpy as np
from array_paths import *

from tiledb.vector_search import _tiledbvspy as vspy

# ctx = tiledb.Ctx()
ctx = vspy.Ctx({})


def test_construct_FeatureVector():
    logging.info(f"siftsmall_ids_uri = {siftsmall_ids_uri}")

    a = vspy.FeatureVector(ctx, siftsmall_ids_uri)
    assert a.feature_type_string() == "uint64"
    assert a.dimension() == 10000


def test_feature_vector_to_numpy():
    a = vspy.FeatureVector(ctx, siftsmall_ids_uri)
    assert a.feature_type_string() == "uint64"
    assert a.dimension() == 10000
    b = np.array(a)
    assert b.ndim == 1
    assert b.shape == (10000,)
    assert b.dtype == np.uint64


def test_numpy_to_feature_vector_array_simple():
    a = np.array(np.random.rand(10000), dtype=np.float32)
    b = vspy.FeatureVector(a)
    assert a.ndim == 1
    logging.info(a.shape)
    assert a.shape[0] == (10000)
    assert b.dimension() == 10000
    assert b.feature_type_string() == "float32"

    c = np.array(b)
    assert c.ndim == 1
    assert c.shape == (10000,)
    assert (a == c).all()


def test_construct_FeatureVectorArray():
    a = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert a.feature_type_string() == "float32"
    assert a.num_vectors() == 10000
    assert a.dimension() == 128

    a = vspy.FeatureVectorArray(ctx, bigann10k_inputs_uri)
    assert a.feature_type_string() == "uint8"
    assert a.num_vectors() == 10000
    assert a.dimension() == 128

    a = vspy.FeatureVectorArray(ctx, fmnistsmall_inputs_uri)
    assert a.feature_type_string() == "float32"
    assert a.num_vectors() == 1000
    assert a.dimension() == 784


def test_feature_vector_array_to_numpy():
    a = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert a.num_vectors() == 10000
    assert a.dimension() == 128
    b = np.array(a)
    assert b.shape == (10000, 128)

    a = vspy.FeatureVectorArray(ctx, bigann10k_inputs_uri)
    assert a.num_vectors() == 10000
    assert a.dimension() == 128
    b = np.array(a)
    assert b.shape == (10000, 128)


def test_numpy_to_feature_vector_array():
    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (10000, 128)
    assert b.dimension() == 128
    assert b.num_vectors() == 10000

    a = np.array(np.random.rand(128, 10000), dtype=np.float32, order="F")
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (128, 10000)
    assert b.dimension() == 128
    assert b.num_vectors() == 10000

    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a.T)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (10000, 128)
    assert b.dimension() == 128
    assert b.num_vectors() == 10000

    a = np.array(np.random.rand(1000000, 128), dtype=np.uint8)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (1000000, 128)
    assert b.dimension() == 128
    assert b.num_vectors() == 1000000

    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))

    c = np.array(b)
    logging.info(c.shape)

    assert a.shape == c.shape
    assert (a == c).all()


def test_construct_IndexFlatL2():
    a = vspy.IndexFlatL2(ctx, siftsmall_inputs_uri)
    assert a.feature_type_string() == "float32"
    assert a.dimension() == 128


def test_query_IndexFlatL2():
    k_nn = 10
    num_queries = 100

    a = vspy.IndexFlatL2(ctx, siftsmall_inputs_uri)
    q = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    gt = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert a.feature_type_string() == "float32"
    assert a.dimension() == 128
    assert q.feature_type_string() == "float32"
    assert q.dimension() == 128
    assert q.num_vectors() == num_queries
    assert gt.feature_type_string() == "uint64"
    assert gt.dimension() == 100
    assert gt.num_vectors() == num_queries

    aq_scores, aq_top_k = a.query(q, k_nn)
    assert aq_top_k.dimension() == k_nn
    assert aq_scores.dimension() == k_nn
    assert aq_top_k.feature_type_string() == "uint64"
    assert aq_scores.feature_type_string() == "float32"
    assert aq_top_k.num_vectors() == num_queries
    assert aq_scores.num_vectors() == num_queries

    logging.info(type(aq_top_k))

    u = np.array(aq_top_k)
    v = np.array(gt)
    logging.info(f"u.shape={u.shape}, v.shape={v.shape}")
    logging.info(f"u.dtype={u.dtype}, v.dtype={v.dtype}")
    assert (u == v[:, 0:k_nn]).all()


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