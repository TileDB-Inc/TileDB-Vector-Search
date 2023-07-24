from common import *

from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.index import IVFFlatIndex
from tiledb.cloud.dag import Mode

import pytest

MINIMUM_ACCURACY = 0.9


@pytest.mark.parametrize("query_type", ["heap", "nth"])
def test_flat_ingestion_u8(tmp_path, query_type):
    dataset_dir = os.path.join(tmp_path, "dataset")
    array_uri = os.path.join(tmp_path, "array")
    create_random_dataset_u8(nb=10000, d=100, nq=100, k=10, path=dataset_dir)
    source_type = "U8BIN"
    dtype = np.uint8
    k = 10

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="FLAT",
        array_uri=array_uri,
        source_uri=os.path.join(dataset_dir, "data"),
        source_type=source_type,
    )
    result = index.query(query_vectors, k=k, query_type=query_type)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


@pytest.mark.parametrize("query_type", ["heap", "nth"])
def test_flat_ingestion_f32(tmp_path, query_type):
    dataset_dir = os.path.join(tmp_path, "dataset")
    array_uri = os.path.join(tmp_path, "array")
    create_random_dataset_f32(nb=10000, d=100, nq=100, k=10, path=dataset_dir)
    source_type = "F32BIN"
    dtype = np.float32
    k = 10

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="FLAT",
        array_uri=array_uri,
        source_uri=os.path.join(dataset_dir, "data"),
        source_type=source_type,
    )
    result = index.query(query_vectors, k=k, query_type=query_type)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_ivf_flat_ingestion_u8(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    array_uri = os.path.join(tmp_path, "array")
    k = 10
    size = 100000
    partitions = 100
    dimensions = 128
    nqueries = 100
    nprobe = 20
    create_random_dataset_u8(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    source_type = "U8BIN"
    dtype = np.uint8

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)
    index = ingest(
        index_type="IVF_FLAT",
        array_uri=array_uri,
        source_uri=os.path.join(dataset_dir, "data"),
        source_type=source_type,
        partitions=partitions,
        input_vectors_per_work_item=int(size / 10),
    )
    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=array_uri, dtype=dtype, memory_budget=int(size / 10))
    result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    result = index_ram.query(
        query_vectors,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    result = index_ram.query(
        query_vectors,
        k=k,
        nprobe=nprobe,
        mode=Mode.LOCAL,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_ivf_flat_ingestion_f32(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    array_uri = os.path.join(tmp_path, "array")
    k = 10
    size = 100000
    dimensions = 128
    partitions = 100
    nqueries = 100
    nprobe = 20

    create_random_dataset_f32(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    source_type = "F32BIN"
    dtype = np.float32

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="IVF_FLAT",
        array_uri=array_uri,
        source_uri=os.path.join(dataset_dir, "data"),
        source_type=source_type,
        partitions=partitions,
        input_vectors_per_work_item=int(size / 10),
    )

    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=array_uri, dtype=dtype, memory_budget=int(size / 10))
    result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    result = index_ram.query(
        query_vectors,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    result = index_ram.query(query_vectors, k=k, nprobe=nprobe, mode=Mode.LOCAL)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_ivf_flat_ingestion_fvec(tmp_path):
    source_uri = "test/data/siftsmall/siftsmall_base.fvecs"
    queries_uri = "test/data/siftsmall/siftsmall_query.fvecs"
    gt_uri = "test/data/siftsmall/siftsmall_groundtruth.ivecs"
    source_type = "FVEC"
    dtype = np.float32
    array_uri = os.path.join(tmp_path, "array")
    k = 100
    dimensions = 128
    partitions = 100
    nqueries = 100
    nprobe = 20

    query_vectors = get_queries_fvec(
        queries_uri, dimensions=dimensions, nqueries=nqueries
    )
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

    index = ingest(
        index_type="IVF_FLAT",
        array_uri=array_uri,
        source_uri=source_uri,
        source_type=source_type,
        partitions=partitions,
    )
    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=array_uri, dtype=dtype)
    result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    result = index_ram.query(
        query_vectors,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    result = index_ram.query(query_vectors, k=k, nprobe=nprobe, mode=Mode.LOCAL)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY
