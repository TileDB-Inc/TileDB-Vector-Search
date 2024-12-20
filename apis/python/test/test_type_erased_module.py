import logging

import numpy as np
from array_paths import *
from common import *

from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search.utils import load_fvecs
from tiledb.vector_search.utils import to_temporal_policy

ctx = vspy.Ctx({})


def test_construct_FeatureVector():
    logging.info(f"siftsmall_ids_uri = {siftsmall_ids_uri}")

    a = vspy.FeatureVector(ctx, siftsmall_ids_uri)
    assert a.feature_type_string() == "uint64"
    assert a.dimensions() == 10000


def test_feature_vector_to_numpy():
    a = vspy.FeatureVector(ctx, siftsmall_ids_uri)
    assert a.feature_type_string() == "uint64"
    assert a.dimensions() == 10000
    b = np.array(a)
    assert b.ndim == 1
    assert b.shape == (10000,)
    assert b.dtype == np.uint64


def test_numpy_to_feature_vector_data_types():
    for dtype in [
        np.float32,
        np.int8,
        np.uint8,
        np.int32,
        np.uint32,
        np.uint64,
    ]:
        if np.issubdtype(dtype, np.integer):
            max_val = np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.floating):
            max_val = np.finfo(dtype).max
        else:
            raise TypeError(f"Unsupported data type {dtype}")

        vector = np.array([max_val], dtype=dtype)
        feature_vector = vspy.FeatureVector(vector)
        assert feature_vector.feature_type_string() == np.dtype(dtype).name
        assert np.array_equal(
            vector, np.array(feature_vector)
        ), f"Arrays were not equal for dtype: {dtype}"


def test_numpy_to_feature_vector_array_simple():
    a = np.array(np.random.rand(10000), dtype=np.float32)
    b = vspy.FeatureVector(a)
    assert a.ndim == 1
    logging.info(a.shape)
    assert a.shape[0] == (10000)
    assert b.dimensions() == 10000
    assert b.feature_type_string() == "float32"

    c = np.array(b)
    assert c.ndim == 1
    assert c.shape == (10000,)
    assert (a == c).all()


def test_construct_FeatureVectorArray():
    a = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert a.feature_type_string() == "float32"
    assert a.num_vectors() == 10000
    assert a.dimensions() == 128

    a = vspy.FeatureVectorArray(ctx, bigann10k_inputs_uri)
    assert a.feature_type_string() == "uint8"
    assert a.num_vectors() == 10000
    assert a.dimensions() == 128

    a = vspy.FeatureVectorArray(ctx, fmnistsmall_inputs_uri)
    assert a.feature_type_string() == "float32"
    assert a.num_vectors() == 1000
    assert a.dimensions() == 784


def test_construct_FeatureVectorArray_with_ids():
    a = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri, siftsmall_ids_uri)
    assert a.feature_type_string() == "float32"
    assert a.num_vectors() == 10000
    assert a.ids_type_string() == "uint64"
    assert a.num_ids() == 10000
    assert a.dimensions() == 128


def test_feature_vector_array_to_numpy():
    a = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert a.num_vectors() == 10000
    assert a.dimensions() == 128
    assert a.num_ids() == 0
    assert a.ids_type_string() == "any"
    b = np.array(a)
    assert b.shape == (10000, 128)

    a = vspy.FeatureVectorArray(ctx, bigann10k_inputs_uri)
    assert a.num_vectors() == 10000
    assert a.dimensions() == 128
    assert a.num_ids() == 0
    assert a.ids_type_string() == "any"
    b = np.array(a)
    assert b.shape == (10000, 128)


def test_numpy_to_feature_vector_array_data_types():
    for dtype in [
        np.float32,
        np.int8,
        np.uint8,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ]:
        for dtype_ids in [np.uint32, np.uint64]:
            if np.issubdtype(dtype, np.integer):
                max_val = np.iinfo(dtype).max
            elif np.issubdtype(dtype, np.floating):
                max_val = np.finfo(dtype).max
            else:
                raise TypeError(f"Unsupported data type {dtype}")

            if np.issubdtype(dtype_ids, np.integer):
                max_val_ids = np.iinfo(dtype_ids).max
            elif np.issubdtype(dtype, np.floating):
                max_val_ids = np.finfo(dtype_ids).max
            else:
                raise TypeError(f"Unsupported ids data type {dtype_ids}")

            vectors = np.array([[max_val]], dtype=dtype)
            ids = np.array([max_val_ids], dtype=dtype_ids)
            feature_vector_array = vspy.FeatureVectorArray(vectors, ids)
            assert feature_vector_array.feature_type_string() == np.dtype(dtype).name
            assert feature_vector_array.ids_type_string() == np.dtype(dtype_ids).name
            assert np.array_equal(
                vectors, np.array(feature_vector_array)
            ), f"Arrays were not equal for dtype: {dtype}, dtype_ids: {dtype_ids}"


def test_numpy_to_feature_vector_array():
    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimensions(), b.num_vectors()))
    assert a.shape == (10000, 128)
    assert b.dimensions() == 128
    assert b.num_vectors() == 10000
    assert a.shape == np.array(b).shape
    assert np.array_equal(a, np.array(b))

    a = np.array(np.random.rand(128, 10000), dtype=np.float32, order="F")
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimensions(), b.num_vectors()))
    assert a.shape == (128, 10000)
    assert b.dimensions() == 128
    assert b.num_vectors() == 10000
    # TODO(paris): This should work, but it doesn't.
    # assert a.shape == np.array(b).shape
    # assert np.array_equal(a, np.array(b))

    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a.T)
    logging.info(a.shape)
    logging.info((b.dimensions(), b.num_vectors()))
    assert a.shape == (10000, 128)
    assert b.dimensions() == 128
    assert b.num_vectors() == 10000
    assert a.shape == np.array(b).shape
    assert np.array_equal(a, np.array(b))

    a = np.array(np.random.rand(1000000, 128), dtype=np.uint8)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimensions(), b.num_vectors()))
    assert a.shape == (1000000, 128)
    assert b.dimensions() == 128
    assert b.num_vectors() == 1000000
    assert a.shape == np.array(b).shape
    assert np.array_equal(a, np.array(b))

    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimensions(), b.num_vectors()))
    assert a.shape == np.array(b).shape
    assert np.array_equal(a, np.array(b))

    a = np.array(np.arange(1, 16, dtype=np.float32).reshape(3, 5), dtype=np.float32)
    assert a.shape == (3, 5)
    assert a.flags.f_contiguous is False
    assert a.flags.c_contiguous is True
    a = np.transpose(a)
    assert a.shape == (5, 3)
    assert a.flags.f_contiguous is True
    assert a.flags.c_contiguous is False
    b = vspy.FeatureVectorArray(a)
    # NOTE(paris): It is strange that we have to transpose this output array to have it match the input array. Should investigate this and fix it.
    assert a.shape == np.transpose(np.array(b)).shape
    assert np.array_equal(a, np.transpose(np.array(b)))

    n = 99
    a = load_fvecs(siftsmall_query_file)[0:n]
    assert a.shape == (n, 128)
    assert a.flags.f_contiguous is False
    assert a.flags.c_contiguous is False
    a = np.transpose(a)
    assert a.shape == (128, n)
    assert a.flags.f_contiguous is False
    assert a.flags.c_contiguous is False
    # NOTE(paris): load_fvecs() returns a view of an array, which is not contiguous, so make it contiguous. Ideally we would handle this in FeatureVectorArray().
    a = np.asfortranarray(a)
    b = vspy.FeatureVectorArray(a)
    # NOTE(paris): It is strange that we have to transpose this output array to have it match the input array. Should investigate this and fix it.
    assert a.shape == np.transpose(np.array(b)).shape
    assert np.array_equal(a, np.transpose(np.array(b)))


def test_numpy_to_feature_vector_array_with_ids():
    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    ids = np.arange(10000, dtype=np.uint64)
    b = vspy.FeatureVectorArray(a, ids)
    assert b.num_ids() == 10000
    assert b.ids_type_string() == "uint64"


def test_numpy_to_feature_vector_single_item():
    a = np.array([1], dtype=np.float32)
    assert a.ndim == 1
    assert a.shape == (1,)
    b = vspy.FeatureVector(a)
    assert b.dimensions() == 1
    assert b.feature_type_string() == "float32"
    c = np.array(b)
    assert c.ndim == 1
    assert c.shape == (1,)
    assert (a == c).all()


def test_TemporalPolicy():
    temporal_policy = vspy.TemporalPolicy()
    assert temporal_policy.timestamp_start() == 0
    assert temporal_policy.timestamp_end() == np.iinfo(np.uint64).max

    temporal_policy = vspy.TemporalPolicy(99)
    assert temporal_policy.timestamp_start() == 0
    assert temporal_policy.timestamp_end() == 99

    temporal_policy = vspy.TemporalPolicy(12, 99)
    assert temporal_policy.timestamp_start() == 12
    assert temporal_policy.timestamp_end() == 99

    temporal_policy = vspy.TemporalPolicy(None, 99)
    assert temporal_policy.timestamp_start() == 0
    assert temporal_policy.timestamp_end() == 99

    temporal_policy = vspy.TemporalPolicy(1, None)
    assert temporal_policy.timestamp_start() == 1
    assert temporal_policy.timestamp_end() == np.iinfo(np.uint64).max


def test_TemporalPolicy_from_timestamp():
    temporal_policy = to_temporal_policy(None)
    assert temporal_policy is None

    temporal_policy = to_temporal_policy(3)
    assert temporal_policy.timestamp_start() == 0
    assert temporal_policy.timestamp_end() == 3

    temporal_policy = to_temporal_policy((0, 33))
    assert temporal_policy.timestamp_start() == 0
    assert temporal_policy.timestamp_end() == 33

    temporal_policy = to_temporal_policy((1, 33))
    assert temporal_policy.timestamp_start() == 1
    assert temporal_policy.timestamp_end() == 33

    temporal_policy = to_temporal_policy((None, 333))
    assert temporal_policy.timestamp_start() == 0
    assert temporal_policy.timestamp_end() == 333

    temporal_policy = to_temporal_policy((3333, None))
    assert temporal_policy.timestamp_start() == 3333
    assert temporal_policy.timestamp_end() == np.iinfo(np.uint64).max

    temporal_policy = to_temporal_policy((None, None))
    assert temporal_policy.timestamp_start() == 0
    assert temporal_policy.timestamp_end() == np.iinfo(np.uint64).max


def test_construct_IndexFlatL2():
    a = vspy.IndexFlatL2(ctx, siftsmall_inputs_uri)
    assert a.feature_type_string() == "float32"
    assert a.dimensions() == 128


def test_query_IndexFlatL2():
    k_nn = 10
    num_queries = 100

    a = vspy.IndexFlatL2(ctx, siftsmall_inputs_uri)
    q = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    gt = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert a.feature_type_string() == "float32"
    assert a.dimensions() == 128

    assert q.dimensions() == 128
    assert q.feature_type_string() == "float32"
    assert q.num_vectors() == num_queries
    assert q.ids_type_string() == "any"
    assert q.num_ids() == 0

    assert gt.dimensions() == 100
    assert gt.feature_type_string() == "uint64"
    assert gt.num_vectors() == num_queries
    assert q.ids_type_string() == "any"
    assert q.num_ids() == 0

    aq_scores, aq_top_k = a.query(q, k_nn)
    assert aq_top_k.dimensions() == k_nn
    assert aq_scores.dimensions() == k_nn
    assert aq_top_k.feature_type_string() == "uint64"
    assert aq_scores.feature_type_string() == "float32"
    assert aq_top_k.num_vectors() == num_queries
    assert aq_scores.num_vectors() == num_queries
    assert aq_top_k.ids_type_string() == "any"
    assert aq_scores.ids_type_string() == "any"
    assert aq_top_k.num_ids() == 0
    assert aq_scores.num_ids() == 0

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
    assert a.dimensions() == 0

    a = vspy.IndexVamana(feature_type="float32")
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "uint32"
    assert a.dimensions() == 0

    a = vspy.IndexVamana(feature_type="uint8", id_type="uint64")
    assert a.feature_type_string() == "uint8"
    assert a.id_type_string() == "uint64"
    assert a.dimensions() == 0

    a = vspy.IndexVamana(feature_type="float32", id_type="int64")
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "int64"
    assert a.dimensions() == 0

    a = vspy.IndexVamana(feature_type="float32", id_type="int64", l_build=11)
    assert a.l_build() == 11

    a = vspy.IndexVamana(feature_type="float32", id_type="int64", r_max_degree=22)
    assert a.r_max_degree() == 22

    a = vspy.IndexVamana(
        feature_type="float32", id_type="int64", l_build=11, r_max_degree=22
    )
    assert a.l_build() == 11
    assert a.r_max_degree() == 22


def test_construct_IndexVamana_with_empty_vector(tmp_path):
    l_search = 100
    k_nn = 10
    index_uri = os.path.join(tmp_path, "array")
    dimensions = 128
    feature_type = "float32"
    id_type = "uint64"
    l_build = 100
    r_max_degree = 101

    # First create an empty index.
    a = vspy.IndexVamana(
        feature_type=feature_type,
        id_type=id_type,
        dimensions=dimensions,
        l_build=l_build,
        r_max_degree=r_max_degree,
    )
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
    assert a.l_build() == l_build
    assert a.r_max_degree() == r_max_degree

    a.train(training_set)

    s, t = a.query(query_set, k_nn, l_search)

    intersections = vspy.count_intersections(t, groundtruth_set, k_nn)
    nt = np.double(t.num_vectors()) * np.double(k_nn)
    recall = intersections / nt
    assert recall == 1.0


def test_inplace_build_query_IndexVamana():
    l_search = 100
    k_nn = 10

    a = vspy.IndexVamana(id_type="uint32", feature_type="float32")

    training_set = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert training_set.feature_type_string() == "float32"
    query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    assert query_set.feature_type_string() == "float32"
    groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert groundtruth_set.feature_type_string() == "uint64"

    a.train(training_set)
    s, t = a.query(query_set, k_nn, l_search)

    intersections = vspy.count_intersections(t, groundtruth_set, k_nn)

    nt = np.double(t.num_vectors()) * np.double(k_nn)
    recall = intersections / nt

    assert recall == 1.0


def test_construct_IndexIVFPQ(tmp_path):
    index_uri = os.path.join(tmp_path, "array")
    if os.path.exists(index_uri):
        shutil.rmtree(index_uri)
    vspy.IndexIVFPQ.create(
        ctx=ctx,
        index_uri=index_uri,
        dimensions=9,
        num_subspaces=3,
        feature_type="float32",
    )
    a = vspy.IndexIVFPQ(ctx, index_uri)
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "uint64"
    assert a.partitioning_index_type_string() == "uint64"
    assert a.dimensions() == 9

    if os.path.exists(index_uri):
        shutil.rmtree(index_uri)
    vspy.IndexIVFPQ.create(
        ctx=ctx,
        index_uri=index_uri,
        dimensions=99,
        num_subspaces=33,
        feature_type="uint8",
        id_type="uint64",
        partitioning_index_type="uint64",
    )
    a = vspy.IndexIVFPQ(ctx, index_uri)
    assert a.feature_type_string() == "uint8"
    assert a.id_type_string() == "uint64"
    assert a.partitioning_index_type_string() == "uint64"
    assert a.dimensions() == 99

    if os.path.exists(index_uri):
        shutil.rmtree(index_uri)
    vspy.IndexIVFPQ.create(
        ctx=ctx,
        index_uri=index_uri,
        dimensions=999,
        num_subspaces=333,
        feature_type="float32",
        id_type="uint32",
        partitioning_index_type="uint64",
    )
    a = vspy.IndexIVFPQ(ctx, index_uri)
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "uint32"
    assert a.partitioning_index_type_string() == "uint64"
    assert a.dimensions() == 999

    build_config_string = vspy.build_config_string()
    assert build_config_string is not None
    assert "tiledb_version" in build_config_string
    logging_string = vspy.logging_string()
    assert logging_string is not None
    assert "Timers" in logging_string


def test_construct_IndexIVFPQ_with_empty_vector(tmp_path):
    index_uri = os.path.join(tmp_path, "array")
    if os.path.exists(index_uri):
        shutil.rmtree(index_uri)

    nprobe = 100
    k_nn = 10
    index_uri = os.path.join(tmp_path, "array")
    dimensions = 128
    partitions = 100
    feature_type = "float32"
    id_type = "uint64"
    partitioning_index_type = "uint64"

    # First create an empty index.
    vspy.IndexIVFPQ.create(
        ctx=ctx,
        index_uri=index_uri,
        dimensions=dimensions,
        feature_type=feature_type,
        id_type=id_type,
        partitioning_index_type=partitioning_index_type,
        num_subspaces=int(dimensions / 2),
    )
    a = vspy.IndexIVFPQ(ctx, index_uri)
    empty_vector = vspy.FeatureVectorArray(dimensions, 0, feature_type, id_type)
    a.train(empty_vector, partitions)
    a.ingest(empty_vector)

    # Then load it again, retrain, and query.
    a = vspy.IndexIVFPQ(ctx, index_uri)
    training_set = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert training_set.feature_type_string() == "float32"
    query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    assert query_set.feature_type_string() == "float32"
    groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert groundtruth_set.feature_type_string() == "uint64"

    a.train(training_set)
    a.ingest(training_set)

    _, ids = a.query(query_set, k_nn, nprobe)
    accuracy = recall(ids, groundtruth_set, k_nn)
    assert accuracy >= 0.87

    b_infinite = vspy.IndexIVFPQ(ctx, index_uri)
    _, ids_infinite = b_infinite.query(query_set, k_nn, nprobe)
    accuracy_infinite = recall(ids_infinite, groundtruth_set, k_nn)
    assert accuracy == accuracy_infinite

    b_finite = vspy.IndexIVFPQ(
        ctx,
        index_uri,
        index_load_strategy=vspy.IndexLoadStrategy.PQ_OOC,
        memory_budget=1000,
    )
    _, ids_finite = b_finite.query(query_set, k_nn, nprobe)
    accuracy_finite = recall(ids_finite, groundtruth_set, k_nn)
    assert accuracy == accuracy_finite


def test_inplace_build_query_IndexIVFPQ(tmp_path):
    index_uri = os.path.join(tmp_path, "array")
    if os.path.exists(index_uri):
        shutil.rmtree(index_uri)

    nprobe = 100
    k_nn = 10
    partitions = 100

    vspy.IndexIVFPQ.create(
        ctx=ctx,
        index_uri=index_uri,
        dimensions=siftsmall_dimensions,
        id_type="uint32",
        partitioning_index_type="uint32",
        feature_type="float32",
        num_subspaces=int(siftsmall_dimensions / 2),
    )
    a = vspy.IndexIVFPQ(ctx, index_uri)

    training_set = vspy.FeatureVectorArray(ctx, siftsmall_inputs_uri)
    assert training_set.feature_type_string() == "float32"
    query_set = vspy.FeatureVectorArray(ctx, siftsmall_query_uri)
    assert query_set.feature_type_string() == "float32"
    groundtruth_set = vspy.FeatureVectorArray(ctx, siftsmall_groundtruth_uri)
    assert groundtruth_set.feature_type_string() == "uint64"

    a.train(training_set, partitions)
    assert a.partitions() == partitions
    a.ingest(training_set)
    assert a.partitions() == partitions

    _, ids = a.query(query_set, k_nn, nprobe)
    accuracy = recall(ids, groundtruth_set, k_nn)
    assert accuracy >= 0.87

    b_infinite = vspy.IndexIVFPQ(ctx, index_uri)
    assert b_infinite.partitions() == partitions
    _, ids_infinite = b_infinite.query(query_set, k_nn, nprobe)
    accuracy_infinite = recall(ids_infinite, groundtruth_set, k_nn)
    assert accuracy == accuracy_infinite

    b_finite = vspy.IndexIVFPQ(
        ctx,
        index_uri,
        index_load_strategy=vspy.IndexLoadStrategy.PQ_OOC,
        memory_budget=999,
    )
    assert b_finite.partitions() == partitions
    _, ids_finite = b_finite.query(query_set, k_nn, nprobe)
    accuracy_finite = recall(ids_finite, groundtruth_set, k_nn)
    assert accuracy == accuracy_finite


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
            assert recall >= 0.998
