import tiledb
import tiledb.vector_search as vs
from tiledb.vector_search import _tiledbvspy as vspy

import numpy as np
import logging

m1_root = "/Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/"
db_uri = m1_root + "sift/sift_base"
centroids_uri = m1_root + "sift/centroids"
parts_uri = m1_root + "sift/parts"
index_uri = m1_root + "sift/index"
sizes_uri = m1_root + "/1M/index_size.tdb"
ids_uri = m1_root + "sift/ids"
query_uri = m1_root + "sift/sift_query"
groundtruth_uri = m1_root + "sift/sift_groundtruth"

bigann1M_base_uri = m1_root + "1M/bigann1M_base"
bigann1M_query_uri = m1_root + "1M/query_public_10k"
bigann1M_groundtruth_uri = m1_root + "1M/bigann_1M_GT_nnids"
bigann1M_centroids_uri = m1_root + "1M/centroids.tdb"
bigann1M_ids_uri = m1_root + "1M/ids.tdb"
bigann1M_index_uri = m1_root + "1M/index.tdb"
bigann1M_index_size_uri = m1_root + "1M/index_size.tdb"
bigann1M_parts_uri = m1_root + "1M/parts.tdb"
bigann1M_flatIVF_index_uri_64_32 = m1_root + "1M/flatIVF_index_1M_base_64_32"

fmnist_train_uri = m1_root + "fmnist/fmnist_train.tdb"
fmnist_test_uri = m1_root + "fmnist/fmnist_test.tdb"
fmnist_groundtruth_uri = m1_root + "fmnist/fmnist_neighbors.tdb"
sift_base_uri = m1_root + "sift/sift_base"

fmnist_distances = m1_root + "fmnist/fmnist_distances.tdb"
fmnist_neighbors = m1_root + "fmnist/fmnist_neighbors.tdb"
fmnist_test = m1_root + "fmnist/fmnist_test.tdb"
fmnist_train = m1_root + "fmnist/fmnist_train.tdb"

diskann_test_256bin = m1_root + "diskann/siftsmall_learn_256pts.fbin"

siftsmall_base_uri = m1_root + "siftsmall/siftsmall_base"
siftsmall_groundtruth_uri = m1_root + "siftsmall/siftsmall_groundtruth"
siftsmall_query_uri = m1_root + "siftsmall/siftsmall_query"
siftsmall_flatIVF_index_uri = m1_root + "siftsmall/flatIVF_index_siftsmall_base"
siftsmall_flatIVF_index_uri_32_64 = m1_root + "siftsmall/flatIVF_index_siftsmall_base_32_64"

# ctx = tiledb.Ctx()
ctx = vspy.Ctx({})


def test_construct_FeatureVector():
    a = vspy.FeatureVector(ctx, ids_uri);
    assert a.feature_type_string() == "uint64"
    assert a.dimension() == 1000000


def test_feature_vector_to_numpy():
    a = vspy.FeatureVector(ctx, ids_uri)
    assert a.feature_type_string() == "uint64"
    assert a.dimension() == 1000000
    b = np.array(a)
    assert b.ndim == 1
    assert b.shape == (1000000,)
    assert b.dtype == np.uint64

def test_numpy_to_feature_vector_array():
    a = np.array(np.random.rand(10000), dtype=np.float32)
    b = vspy.FeatureVector(a)
    assert a.ndim == 1
    logging.info(a.shape)
    assert a.shape == (10000)
    assert b.dimension() == 10000
    assert b.feature_type_string() == "float32"

    c = np.array(b)
    assert c.ndim == 1
    assert c.shape == (10000,)
    assert (a == c).all()

def test_construct_FeatureVectorArray():
    a = vspy.FeatureVectorArray(ctx, siftsmall_base_uri)
    assert a.feature_type_string() == "float32"
    assert a.num_vectors() == 10000
    assert a.dimension() == 128

    a = vspy.FeatureVectorArray(ctx, bigann1M_base_uri)
    assert a.feature_type_string() == "uint8"
    assert a.num_vectors() == 1000000
    assert a.dimension() == 128


def test_feature_vector_array_to_numpy():
    a = vspy.FeatureVectorArray(ctx, siftsmall_base_uri)
    assert a.num_vectors() == 10000
    assert a.dimension() == 128
    b = np.array(a)
    assert b.shape == (10000, 128)

    a = vspy.FeatureVectorArray(ctx, bigann1M_base_uri)
    assert a.num_vectors() == 1000000
    assert a.dimension() == 128
    b = np.array(a)
    assert b.shape == (1000000, 128)


def test_numpy_to_feature_vector_array():
    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (10000, 128)
    assert b.dimension() == 128
    assert b.num_vectors() == 10000

    a = np.array(np.random.rand(128, 10000), dtype=np.float32, order='F')
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


#    logging.info(a[1:5, 1:5])
#    logging.info(c[1:5, 1:5])

def test_construct_IndexFlatL2():
    a = vspy.IndexFlatL2(ctx, siftsmall_base_uri)
    assert a.feature_type_string() == "float32"
    assert a.dimension() == 128

def test_query_IndexFlatL2():
    k_nn = 10
    num_queries = 100

    a = vspy.IndexFlatL2(ctx, siftsmall_base_uri)
    q = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    gt = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert a.feature_type_string() == "float32"
    assert a.dimension() == 128
    assert q.feature_type_string() == "float32"
    assert q.dimension() == 128
    assert q.num_vectors() == num_queries
    assert gt.feature_type_string() == "int32"
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
    assert (u == v[:,0:k_nn]).all()


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

    a = vspy.IndexIVFFlat(ctx, siftsmall_flatIVF_index_uri_32_64)
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "uint32"
    assert a.px_type_string() == "uint64"
    assert a.dimension() == 128

    a = vspy.IndexIVFFlat(ctx, bigann1M_flatIVF_index_uri_64_32)
    assert a.feature_type_string() == "uint8"
    assert a.id_type_string() == "uint64"
    assert a.px_type_string() == "uint32"
    assert a.dimension() == 128

def test_inplace_build_infinite_query_IndexIVFFlat():
    k_nn = 10
    nprobe = 32
    num_queries = 100

    for nprobe in [ 8, 32]:
        a = vspy.IndexIVFFlat(id_type="uint32", px_type="uint32")

        training_set = vspy.FeatureVectorArray(ctx, siftsmall_base_uri)
        assert training_set.feature_type_string() == "float32"

        query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
        assert query_set.feature_type_string() == "float32"

        groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
        assert groundtruth_set.feature_type_string() == "int32"

        a.train(training_set, "random");
        a.add(training_set);
        s, t = a.query_infinite_ram(query_set, k_nn, nprobe)

        intersections = vspy.count_intersections(t, groundtruth_set, k_nn);

        nt = np.double(t.num_vectors()) * np.double(k_nn);
        recall = intersections / nt

        logging.info(f"nprobe = {nprobe}, recall={recall}")

        if (nprobe == 8):
            assert recall > 0.95
        if (nprobe == 32):
            assert recall == 1.0

def test_read_index_and_infinite_query():
    k_nn = 10
    nprobe = 32
    num_queries = 100

    for nprobe in [ 8, 32]:
        a = vspy.IndexIVFFlat(ctx, siftsmall_flatIVF_index_uri_32_64)
        assert a.feature_type_string() == "float32"
        assert a.id_type_string() == "uint32"
        assert a.px_type_string() == "uint64"
        assert a.dimension() == 128

        query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
        assert query_set.feature_type_string() == "float32"

        groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
        assert groundtruth_set.feature_type_string() == "int32"

        s, t = a.query_infinite_ram(query_set, k_nn, nprobe)

        intersections = vspy.count_intersections(t, groundtruth_set, k_nn);

        nt = np.double(t.num_vectors()) * np.double(k_nn);
        recall = intersections / nt

        logging.info(f"nprobe = {nprobe}, recall={recall}")

        if (nprobe == 8):
            assert recall > 0.925
        if (nprobe == 32):
            assert recall == 1.0

def test_read_index_and_finite_query_default_upper_bound():
    k_nn = 10
    nprobe = 32
    num_queries = 100

    for nprobe in [ 8, 32]:
        a = vspy.IndexIVFFlat(ctx, siftsmall_flatIVF_index_uri_32_64)
        query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
        groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)

        s, t = a.query_finite_ram(query_set, k_nn, nprobe)

        intersections = vspy.count_intersections(t, groundtruth_set, k_nn);

        nt = np.double(t.num_vectors()) * np.double(k_nn);
        recall = intersections / nt

        logging.info(f"nprobe = {nprobe}, recall={recall}")

        if (nprobe == 8):
            assert recall > 0.925
        if (nprobe == 32):
            assert recall == 1.0


def test_read_index_and_finite_query_0_upper_bound():
    k_nn = 10
    nprobe = 32
    num_queries = 100

    for nprobe in [ 8, 32]:
        a = vspy.IndexIVFFlat(ctx, siftsmall_flatIVF_index_uri_32_64)
        query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
        groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)

        s, t = a.query_finite_ram(query_set, k_nn, nprobe, 0)

        intersections = vspy.count_intersections(t, groundtruth_set, k_nn);

        nt = np.double(t.num_vectors()) * np.double(k_nn);
        recall = intersections / nt

        logging.info(f"nprobe = {nprobe}, recall={recall}")

        if (nprobe == 8):
            assert recall > 0.925
        if (nprobe == 32):
            assert recall == 1.0


def test_read_index_and_finite_query_1000_upper_bound():
    k_nn = 10
    nprobe = 32
    num_queries = 100

    for nprobe in [ 8, 32]:
        a = vspy.IndexIVFFlat(ctx, siftsmall_flatIVF_index_uri_32_64)
        query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
        groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)

        s, t = a.query_finite_ram(query_set, k_nn, nprobe, 1000)

        intersections = vspy.count_intersections(t, groundtruth_set, k_nn);

        nt = np.double(t.num_vectors()) * np.double(k_nn);
        recall = intersections / nt

        logging.info(f"nprobe = {nprobe}, recall={recall}")

        if (nprobe == 8):
            assert recall > 0.925
        if (nprobe == 32):
            assert recall == 1.0