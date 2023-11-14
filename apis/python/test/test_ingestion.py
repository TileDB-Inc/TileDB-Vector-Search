import numpy as np

from common import *

from tiledb.vector_search.utils import load_fvecs
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.flat_index import FlatIndex
from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
from tiledb.vector_search.module import array_to_matrix, kmeans_fit, kmeans_predict
from tiledb.cloud.dag import Mode

MINIMUM_ACCURACY = 0.85
MAX_UINT64 = np.iinfo(np.dtype("uint64")).max


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
    _, result = index.query(query_vectors, k=k)
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
    _, result = index.query(query_vectors, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = FlatIndex(uri=index_uri)
    _, result = index_ram.query(query_vectors, k=k)
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
    _, result = index.query(query_vectors, k=k)
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
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri, memory_budget=int(size / 10))
    _, result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(
        query_vectors,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(
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

    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri)
    _, result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri, memory_budget=int(size / 10))
    _, result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(
        query_vectors,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(query_vectors, k=k, nprobe=nprobe, mode=Mode.LOCAL)
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
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    # Test single query vector handling
    _, result1 = index.query(query_vectors[10], k=k, nprobe=nprobe)
    assert accuracy(result1, np.array([gt_i[10]])) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri)
    _, result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(
        query_vectors,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    # NB: local mode currently does not return distances
    _, result = index_ram.query(query_vectors, k=k, nprobe=nprobe, mode=Mode.LOCAL)
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
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    # Test single query vector handling
    _, result1 = index.query(query_vectors[10], k=k, nprobe=nprobe)
    assert accuracy(result1, np.array([gt_i[10]])) > MINIMUM_ACCURACY

    index_ram = IVFFlatIndex(uri=index_uri)
    _, result = index_ram.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(
        query_vectors,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(query_vectors, k=k, nprobe=nprobe, mode=Mode.LOCAL)
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
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i, external_ids_offset) > MINIMUM_ACCURACY


def test_ivf_flat_ingestion_with_updates(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    k = 10
    size = 100000
    partitions = 100
    dimensions = 128
    nqueries = 100
    nprobe = 20
    data = create_random_dataset_u8(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
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
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    update_ids_offset = MAX_UINT64-size
    updated_ids = {}
    for i in range(100):
        index.delete(external_id=i)
        index.update(vector=data[i].astype(dtype), external_id=i + update_ids_offset)
        updated_ids[i + update_ids_offset] = i

    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i, updated_ids=updated_ids) > MINIMUM_ACCURACY

    index = index.consolidate_updates()
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i, updated_ids=updated_ids) > MINIMUM_ACCURACY

def test_ivf_flat_ingestion_with_batch_updates(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    k = 10
    size = 100000
    partitions = 100
    dimensions = 128
    nqueries = 100
    nprobe = 20
    data = create_random_dataset_u8(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
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
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    update_ids = {}
    updated_ids = {}
    update_ids_offset = MAX_UINT64 - size
    for i in range(0, 100000, 2):
        update_ids[i] = i + update_ids_offset
        updated_ids[i + update_ids_offset] = i
    external_ids = np.zeros((len(update_ids) * 2), dtype=np.uint64)
    updates = np.empty((len(update_ids) * 2), dtype='O')
    id = 0
    for prev_id, new_id in update_ids.items():
        external_ids[id] = prev_id
        updates[id] = np.array([], dtype=dtype)
        id += 1
        external_ids[id] = new_id
        updates[id] = data[prev_id].astype(dtype)
        id += 1

    index.update_batch(vectors=updates, external_ids=external_ids)
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i, updated_ids=updated_ids) > MINIMUM_ACCURACY

    index = index.consolidate_updates()
    _, result = index.query(query_vectors, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i, updated_ids=updated_ids) > MINIMUM_ACCURACY



def test_kmeans():
    k = 128
    d = 16
    n = k * k
    max_iter = 16
    n_init = 10
    verbose = False

    import sklearn.model_selection
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    X, _, centers = make_blobs(n_samples=n, n_features=d, centers=k, return_centers=True, random_state=1)
    X = X.astype("float32")

    data, queries = sklearn.model_selection.train_test_split(
        X, test_size=0.1, random_state=1
    )

    data_x = np.array([[1.0573647,   5.082087],
                      [-6.229642,   -1.3590931],
                      [0.7446737,    6.3828287],
                      [-7.698864,   -3.0493321],
                      [2.1362762,   -4.4448104],
                      [1.04019,     -4.0389647],
                      [0.38996044,   5.7235265],
    [1.7470839,  -4.717076]]).astype("float32")
    queries_x = np.array([[-7.3712273, -1.1178735]]).astype("float32")

    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, verbose=verbose, init="random", random_state=1)
    km.fit(data)
    centroids_sk = km.cluster_centers_
    results_sk = km.predict(queries)

    centroids_tdb = kmeans_fit(
        k, "random", max_iter, verbose, n_init, array_to_matrix(np.transpose(data)), seed=1
    )
    centroids_tdb_np = np.transpose(np.array(centroids_tdb))
    results_tdb = kmeans_predict(centroids_tdb, array_to_matrix(np.transpose(queries)))
    results_tdb_np = np.transpose(np.array(results_tdb))

    def get_score(centroids, results):
        x = []
        for i in range(len(queries)):
            x.append(np.linalg.norm(queries[i] - centroids[results[i]]))
        return np.mean(np.array(x))

    sklearn_score = get_score(centroids_sk, results_sk)
    tdb_score = get_score(centroids_tdb_np, results_tdb_np)
    print("Random initialization:")
    print(f"sklearn score: {sklearn_score}")
    print(f"tiledb score: {tdb_score}")

    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, verbose=verbose, init="k-means++", random_state=1)
    km.fit(data)
    centroids_sk = km.cluster_centers_
    results_sk = km.predict(queries)

    assert tdb_score < 1.5 * sklearn_score

    centroids_tdb = kmeans_fit(
        k, "k-means++", max_iter, verbose, n_init, array_to_matrix(np.transpose(data)), seed=1
    )
    centroids_tdb_np = np.transpose(np.array(centroids_tdb))
    results_tdb = kmeans_predict(centroids_tdb, array_to_matrix(np.transpose(queries)))
    results_tdb_np = np.transpose(np.array(results_tdb))

    sklearn_score = get_score(centroids_sk, results_sk)
    tdb_score = get_score(centroids_tdb_np, results_tdb_np)
    print("K-means++:")
    print(f"sklearn score: {sklearn_score}")
    print(f"tiledb score: {tdb_score}")

    assert tdb_score < 1.5 * sklearn_score

test_kmeans()