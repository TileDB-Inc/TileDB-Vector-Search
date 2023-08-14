from common import *

from tiledb.vector_search.utils import load_fvecs
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.flat_index import FlatIndex
from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
from tiledb.cloud.dag import Mode

MINIMUM_ACCURACY = 0.85


def test_flat_ingestion_u8(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    create_random_dataset_u8(nb=10000, d=100, nq=100, k=10, path=dataset_dir)
    dtype = np.uint8
    k = 10

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
    )
    result = index.query(query_vectors, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_flat_ingestion_f32(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    create_random_dataset_f32(nb=10000, d=100, nq=100, k=10, path=dataset_dir)
    dtype = np.float32
    k = 10

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.f32bin"),
    )
    result = index.query(query_vectors, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = FlatIndex(uri=index_uri)
    result = index_ram.query(query_vectors, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_flat_ingestion_external_id_u8(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    size = 10000
    dtype = np.uint8
    create_random_dataset_u8(nb=size, d=100, nq=100, k=10, path=dataset_dir)
    k = 10
    external_ids_offset = 100

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)
    external_ids = np.array([range(external_ids_offset, size+external_ids_offset)], np.uint64)

    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
        external_ids=external_ids
    )
    result = index.query(query_vectors, k=k)
    assert accuracy(result, gt_i, external_ids_offset=external_ids_offset) > MINIMUM_ACCURACY


def test_ivf_flat_ingestion_u8(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    k = 10
    size = 100000
    partitions = 100
    dimensions = 128
    nqueries = 100
    nprobe = 20
    create_random_dataset_u8(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    dtype = np.uint8

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
        partitions=partitions,
        input_vectors_per_work_item=int(size / 10),
    )
    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri, memory_budget=int(size / 10))
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
    index_uri = os.path.join(tmp_path, "array")
    k = 10
    size = 100000
    dimensions = 128
    partitions = 100
    nqueries = 100
    nprobe = 20

    create_random_dataset_f32(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    dtype = np.float32

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.f32bin"),
        partitions=partitions,
        input_vectors_per_work_item=int(size / 10),
    )

    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri)
    result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri, memory_budget=int(size / 10))
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
    index_uri = os.path.join(tmp_path, "array")
    k = 100
    partitions = 100
    nqueries = 100
    nprobe = 20

    query_vectors = load_fvecs(queries_uri)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=source_uri,
        partitions=partitions,
    )
    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    # Test single query vector handling
    result1 = index.query(query_vectors[10], k=k, nprobe=nprobe)
    assert accuracy(result1, np.array([gt_i[10]])) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri)
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


def test_ivf_flat_ingestion_numpy(tmp_path):
    source_uri = "test/data/siftsmall/siftsmall_base.fvecs"
    queries_uri = "test/data/siftsmall/siftsmall_query.fvecs"
    gt_uri = "test/data/siftsmall/siftsmall_groundtruth.ivecs"
    index_uri = os.path.join(tmp_path, "array")
    k = 100
    partitions = 100
    nqueries = 100
    nprobe = 20

    input_vectors = load_fvecs(source_uri)

    query_vectors = load_fvecs(queries_uri)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=input_vectors,
        partitions=partitions,
    )
    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    # Test single query vector handling
    result1 = index.query(query_vectors[10], k=k, nprobe=nprobe)
    assert accuracy(result1, np.array([gt_i[10]])) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri)
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
def test_ivf_flat_ingestion_external_ids_numpy(tmp_path):
    source_uri = "test/data/siftsmall/siftsmall_base.fvecs"
    queries_uri = "test/data/siftsmall/siftsmall_query.fvecs"
    gt_uri = "test/data/siftsmall/siftsmall_groundtruth.ivecs"
    index_uri = os.path.join(tmp_path, "array")
    k = 100
    partitions = 100
    nqueries = 100
    nprobe = 20
    size = 10000
    external_ids_offset = 100

    input_vectors = load_fvecs(source_uri)

    query_vectors = load_fvecs(queries_uri)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)
    external_ids = np.array([range(external_ids_offset, size+external_ids_offset)], np.uint64)
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=input_vectors,
        partitions=partitions,
        external_ids=external_ids
    )
    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i, external_ids_offset) > MINIMUM_ACCURACY

def test_ivf_flat_ingestion_with_updates(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    additions_uri = os.path.join(tmp_path, "additions")
    deletes_uri = os.path.join(tmp_path, "deletes")
    k = 10
    size = 100000
    partitions = 100
    dimensions = 128
    nqueries = 100
    nprobe = 20
    data = create_random_dataset_u8(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    dtype = np.uint8

    external_id_dim = tiledb.Dim(
        name="external_id", domain=(0, 2**63-1), dtype=np.dtype(np.uint64)
    )
    dom = tiledb.Domain(external_id_dim)
    vector_attr = tiledb.Attr(name="vector", dtype=np.dtype(dtype), var=True)
    delete_attr = tiledb.Attr(name="delete", dtype=np.dtype(np.uint8))
    additions_schema = tiledb.ArraySchema(
        domain=dom,
        sparse=True,
        attrs=[vector_attr],
    )
    tiledb.Array.create(additions_uri, additions_schema)
    deletes_schema = tiledb.ArraySchema(
        domain=dom,
        sparse=True,
        attrs=[delete_attr],
    )
    tiledb.Array.create(deletes_uri, deletes_schema)
    additions_array = tiledb.open(additions_uri, mode="w")
    deletes_array = tiledb.open(deletes_uri, mode="w")
    update_ids = {}
    updated_ids = {}
    for i in range(0, 100000, 2):
        update_ids[i] = i + 1000000
        updated_ids[i + 1000000] = i
    delete_ids = np.zeros((len(update_ids)), dtype=np.uint64)
    addition_ids = np.zeros((len(update_ids)), dtype=np.uint64)
    deletes = np.zeros((len(update_ids)), dtype=np.uint8)
    additions = np.empty((len(update_ids)), dtype='O')
    id = 0
    for prev_id, new_id in update_ids.items():
        delete_ids[id] = prev_id
        deletes[id] = 0
        addition_ids[id] = new_id
        additions[id] = data[prev_id].astype(dtype)
        id += 1
    deletes_array[delete_ids] = {"delete": deletes}
    additions_array[addition_ids] = {"vector": additions}
    additions_array.close()
    deletes_array.close()

    query_vectors = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
        partitions=partitions,
        input_vectors_per_work_item=int(size / 10),
        deletes_uri=deletes_uri,
        additions_uri=additions_uri
    )
    result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i, updated_ids=updated_ids) > MINIMUM_ACCURACY
