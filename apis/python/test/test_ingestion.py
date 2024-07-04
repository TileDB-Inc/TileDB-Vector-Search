import time

import numpy as np
import pytest
from array_paths import *
from common import *
from common import load_metadata

from tiledb.cloud.dag import Mode
from tiledb.vector_search import _tiledbvspy as vspy
from tiledb.vector_search.index import Index
from tiledb.vector_search.ingestion import TrainingSamplingPolicy
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
from tiledb.vector_search.module import array_to_matrix
from tiledb.vector_search.module import kmeans_fit
from tiledb.vector_search.module import kmeans_predict
from tiledb.vector_search.utils import MAX_UINT64
from tiledb.vector_search.utils import is_type_erased_index
from tiledb.vector_search.utils import load_fvecs
from tiledb.vector_search.utils import metadata_to_list

MINIMUM_ACCURACY = 0.85
MINIMUM_ACCURACY_IVF_PQ = 0.75


def query_and_check_equals(index, queries, expected_result_d, expected_result_i):
    result_d, result_i = index.query(queries, k=1)
    check_equals(
        result_d=result_d,
        result_i=result_i,
        expected_result_d=expected_result_d,
        expected_result_i=expected_result_i,
    )


def test_vamana_ingestion_u8(tmp_path):
    vfs = tiledb.VFS()

    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    if os.path.exists(index_uri):
        shutil.rmtree(index_uri)

    l_build = 101
    r_max_degree = 65
    dimensions = 100
    create_random_dataset_u8(nb=10000, d=dimensions, nq=100, k=10, path=dataset_dir)
    dtype = np.dtype(np.uint8)
    k = 10

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="VAMANA",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
        l_build=l_build,
        r_max_degree=r_max_degree,
    )

    # This is not a public API, but we directly load the C++ type-erased index to test it. If you
    # are a library user, you should not do this yourself, as the API may change.
    ctx = vspy.Ctx({})
    type_erased_index = vspy.IndexVamana(ctx, index_uri, None)
    assert type_erased_index.dimensions() == dimensions
    assert type_erased_index.l_build() == l_build
    assert type_erased_index.r_max_degree() == r_max_degree

    _, result = index.query(queries, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_uri = move_local_index_to_new_location(index_uri)
    index_ram = VamanaIndex(uri=index_uri)
    _, result = index_ram.query(queries, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    assert vfs.dir_size(index_uri) > 0
    Index.delete_index(uri=index_uri, config={})
    assert vfs.dir_size(index_uri) == 0


def test_flat_ingestion_u8(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    create_random_dataset_u8(nb=10000, d=100, nq=100, k=10, path=dataset_dir)
    dtype = np.uint8
    k = 10

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
    )
    _, result = index.query(queries, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_uri = move_local_index_to_new_location(index_uri)
    index_ram = FlatIndex(uri=index_uri)
    _, result = index_ram.query(queries, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_flat_ingestion_f32(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    create_random_dataset_f32(nb=10000, d=100, nq=100, k=10, path=dataset_dir)
    dtype = np.float32
    k = 10

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.f32bin"),
    )
    _, result = index.query(queries, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_uri = move_local_index_to_new_location(index_uri)
    index_ram = FlatIndex(uri=index_uri)
    _, result = index_ram.query(queries, k=k)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_flat_ingestion_external_id_u8(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    size = 10000
    dtype = np.uint8
    create_random_dataset_u8(nb=size, d=100, nq=100, k=10, path=dataset_dir)
    k = 10
    external_ids_offset = 100

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)
    external_ids = np.array(
        [range(external_ids_offset, size + external_ids_offset)], np.uint64
    )

    index = ingest(
        index_type="FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
        external_ids=external_ids,
    )
    _, result = index.query(queries, k=k)
    assert (
        accuracy(result, gt_i, external_ids_offset=external_ids_offset)
        > MINIMUM_ACCURACY
    )

    index_uri = move_local_index_to_new_location(index_uri)
    index_ram = FlatIndex(uri=index_uri)
    _, result = index_ram.query(queries, k=k)
    assert (
        accuracy(result, gt_i, external_ids_offset=external_ids_offset)
        > MINIMUM_ACCURACY
    )


def test_ivf_flat_ingestion_u8(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    k = 10
    size = 10000
    partitions = 100
    dimensions = 129
    nqueries = 100
    nprobe = 20
    create_random_dataset_u8(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    dtype = np.uint8

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
        partitions=partitions,
        input_vectors_per_work_item=int(size / 10),
    )
    _, result = index.query(queries, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    index_uri = move_local_index_to_new_location(index_uri)
    index_ram = IVFFlatIndex(uri=index_uri, memory_budget=int(size / 10))
    _, result = index_ram.query(queries, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(
        queries,
        k=k,
        nprobe=nprobe,
        use_nuv_implementation=True,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY

    _, result = index_ram.query(
        queries,
        k=k,
        nprobe=nprobe,
        mode=Mode.LOCAL,
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_ivf_flat_ingestion_f32(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    k = 10
    size = 10000
    dimensions = 127
    partitions = 100
    nqueries = 100
    nprobe = 20
    num_subspaces = 127

    create_random_dataset_f32(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    dtype = np.float32

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        minimum_accuracy = (
            MINIMUM_ACCURACY_IVF_PQ if index_type == "IVF_PQ" else MINIMUM_ACCURACY
        )
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=os.path.join(dataset_dir, "data.f32bin"),
            partitions=partitions,
            input_vectors_per_work_item=int(size / 10),
            num_subspaces=num_subspaces,
        )

        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        index_uri = move_local_index_to_new_location(index_uri)
        index_ram = index_class(uri=index_uri)
        _, result = index_ram.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        index_ram = index_class(uri=index_uri, memory_budget=int(size / 10))
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


def test_ingestion_fvec(tmp_path):
    vfs = tiledb.VFS()

    source_uri = siftsmall_inputs_file
    queries_uri = siftsmall_query_file
    gt_uri = siftsmall_groundtruth_file
    k = 100
    partitions = 100
    nqueries = 100
    nprobe = 20
    num_subspaces = 64
    queries = load_fvecs(queries_uri)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        minimum_accuracy = (
            MINIMUM_ACCURACY_IVF_PQ if index_type == "IVF_PQ" else MINIMUM_ACCURACY
        )
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=source_uri,
            partitions=partitions,
            num_subspaces=num_subspaces,
        )
        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        index_uri = move_local_index_to_new_location(index_uri)
        index_ram = index_class(uri=index_uri)
        _, result = index_ram.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        _, result = index_ram.query(
            queries,
            k=k,
            nprobe=nprobe,
            use_nuv_implementation=True,
        )
        assert accuracy(result, gt_i) > minimum_accuracy

        # NB: local mode currently does not return distances
        _, result = index_ram.query(queries, k=k, nprobe=nprobe, mode=Mode.LOCAL)
        assert accuracy(result, gt_i) > minimum_accuracy

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ingestion_numpy(tmp_path):
    vfs = tiledb.VFS()

    source_uri = siftsmall_inputs_file
    queries_uri = siftsmall_query_file
    gt_uri = siftsmall_groundtruth_file
    k = 100
    partitions = 100
    nqueries = 100
    nprobe = 20
    num_subspaces = siftsmall_dimensions / 2

    input_vectors = load_fvecs(source_uri)

    queries = load_fvecs(queries_uri)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        minimum_accuracy = (
            MINIMUM_ACCURACY_IVF_PQ if index_type == "IVF_PQ" else MINIMUM_ACCURACY
        )
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            input_vectors=input_vectors,
            partitions=partitions,
            num_subspaces=num_subspaces,
        )
        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        index_uri = move_local_index_to_new_location(index_uri)
        index_ram = index_class(uri=index_uri)
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

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ingestion_numpy_i8(tmp_path):
    vfs = tiledb.VFS()

    source_uri = siftsmall_inputs_file
    queries_uri = siftsmall_query_file
    gt_uri = siftsmall_groundtruth_file
    index_uri = os.path.join(tmp_path, "array")
    k = 100
    partitions = 100
    nqueries = 100
    nprobe = 20
    num_subspaces = siftsmall_dimensions / 2

    input_vectors = quantize_embeddings_int8(load_fvecs(source_uri))

    queries = quantize_embeddings_int8(load_fvecs(queries_uri)).astype(np.float32)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        minimum_accuracy = (
            MINIMUM_ACCURACY_IVF_PQ if index_type == "IVF_PQ" else MINIMUM_ACCURACY
        )

        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            input_vectors=input_vectors,
            partitions=partitions,
            num_subspaces=num_subspaces,
        )
        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        index_uri = move_local_index_to_new_location(index_uri)
        index_ram = index_class(uri=index_uri)
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

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ingestion_multiple_workers(tmp_path):
    vfs = tiledb.VFS()

    source_uri = siftsmall_inputs_file
    queries_uri = siftsmall_query_file
    gt_uri = siftsmall_groundtruth_file
    k = 100
    partitions = 100
    nqueries = 100
    nprobe = 20
    num_subspaces = siftsmall_dimensions / 2

    queries = load_fvecs(queries_uri)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        minimum_accuracy = (
            MINIMUM_ACCURACY_IVF_PQ if index_type == "IVF_PQ" else MINIMUM_ACCURACY
        )
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=source_uri,
            partitions=partitions,
            input_vectors_per_work_item=421,
            max_tasks_per_stage=4,
            num_subspaces=num_subspaces,
        )
        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        index_uri = move_local_index_to_new_location(index_uri)
        index_ram = index_class(uri=index_uri)
        _, result = index_ram.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        _, result = index_ram.query(
            queries,
            k=k,
            nprobe=nprobe,
            use_nuv_implementation=True,
        )
        assert accuracy(result, gt_i) > minimum_accuracy

        # NB: local mode currently does not return distances
        _, result = index_ram.query(queries, k=k, nprobe=nprobe, mode=Mode.LOCAL)
        assert accuracy(result, gt_i) > minimum_accuracy

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ingestion_external_ids_numpy(tmp_path):
    vfs = tiledb.VFS()

    source_uri = siftsmall_inputs_file
    queries_uri = siftsmall_query_file
    gt_uri = siftsmall_groundtruth_file
    k = 100
    partitions = 100
    nqueries = 100
    nprobe = 20
    size = 10000
    num_subspaces = siftsmall_dimensions / 2
    external_ids_offset = 100

    input_vectors = load_fvecs(source_uri)

    queries = load_fvecs(queries_uri)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)
    external_ids = np.array(
        [range(external_ids_offset, size + external_ids_offset)], np.uint64
    )

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        minimum_accuracy = (
            MINIMUM_ACCURACY_IVF_PQ if index_type == "IVF_PQ" else MINIMUM_ACCURACY
        )
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            input_vectors=input_vectors,
            partitions=partitions,
            external_ids=external_ids,
            num_subspaces=num_subspaces,
        )
        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i, external_ids_offset) > minimum_accuracy

        index_uri = move_local_index_to_new_location(index_uri)
        index_ram = index_class(uri=index_uri)
        _, result = index_ram.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i, external_ids_offset) > minimum_accuracy

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ingestion_timetravel(tmp_path):
    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        index_uri = os.path.join(tmp_path, f"array_{index_type}")

        data = np.array([[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]], dtype=np.float32)
        default_result_d = [[np.finfo(np.float32).max], [np.finfo(np.float32).max]]
        default_result_i = [[np.iinfo(np.uint64).max], [np.iinfo(np.uint64).max]]

        # We ingest at timestamp 10.
        ingest(
            index_type=index_type,
            index_uri=index_uri,
            input_vectors=data,
            index_timestamp=10,
            num_subspaces=2,
        )

        # If we load the index with any timestamp < 10, then we have no data and so have no results.
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=0),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=9),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(5, 9)),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(None, 9)),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )

        # If we load the index with timestamp >= 10 then we get results.
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=10),
            queries=data,
            expected_result_d=[[0], [0]],
            expected_result_i=[[0], [1]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=1000),
            queries=data,
            expected_result_d=[[0], [0]],
            expected_result_i=[[0], [1]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(5, 15)),
            queries=data,
            expected_result_d=[[0], [0]],
            expected_result_i=[[0], [1]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(None, 20)),
            queries=data,
            expected_result_d=[[0], [0]],
            expected_result_i=[[0], [1]],
        )

        # We add a third vector at timestamp 20 and consolidate updates, meaning we'll re-ingest at timestamp = 20.
        data = np.array(
            [[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3], [3.0, 3.1, 3.2, 3.3]],
            dtype=np.float32,
        )
        default_result_d = [
            [np.finfo(np.float32).max],
            [np.finfo(np.float32).max],
            [np.finfo(np.float32).max],
        ]
        default_result_i = [
            [np.iinfo(np.uint64).max],
            [np.iinfo(np.uint64).max],
            [np.iinfo(np.uint64).max],
        ]
        index = index_class(uri=index_uri)
        index.update(
            vector=data[2],
            external_id=2,
            timestamp=20,
        )

        if index_type == "IVF_PQ":
            # TODO(SC-48888): Fix consolidation for IVF_PQ.
            continue
        index = index.consolidate_updates()

        # We still have no results before timestamp 10.
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=0),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=9),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(5, 9)),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(None, 9)),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )

        # We have no results if we load in between 10 and 20.
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(11, 19)),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )

        # If we load the index from timestamp 0 -> 19, we only are returned the first two vectors.
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=10),
            queries=data,
            expected_result_d=[[0], [0], [4]],
            expected_result_i=[[0], [1], [1]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=19),
            queries=data,
            expected_result_d=[[0], [0], [4]],
            expected_result_i=[[0], [1], [1]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(0, 19)),
            queries=data,
            expected_result_d=[[0], [0], [4]],
            expected_result_i=[[0], [1], [1]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(None, 19)),
            queries=data,
            expected_result_d=[[0], [0], [4]],
            expected_result_i=[[0], [1], [1]],
        )

        # But if we load with timestamp >= 20 then we get results for all three vectors.
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=None),
            queries=data,
            expected_result_d=[[0], [0], [0]],
            expected_result_i=[[0], [1], [2]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=1000),
            queries=data,
            expected_result_d=[[0], [0], [0]],
            expected_result_i=[[0], [1], [2]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(0, 1000)),
            queries=data,
            expected_result_d=[[0], [0], [0]],
            expected_result_i=[[0], [1], [2]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(None, 20)),
            queries=data,
            expected_result_d=[[0], [0], [0]],
            expected_result_i=[[0], [1], [2]],
        )

        with tiledb.Group(index_uri, "r") as group:
            assert metadata_to_list(group, "ingestion_timestamps") == [10, 21]
            assert metadata_to_list(group, "base_sizes") == [2, 3]
            assert group.meta["has_updates"] == 1
            if index_type != "VAMANA":
                assert metadata_to_list(group, "partition_history") == [1, 1]
            if index_type == "VAMANA":
                num_edges_history = metadata_to_list(group, "num_edges_history")
                assert len(num_edges_history) == 2
                second_num_edges = num_edges_history[1]

        # Clear all history at timestamp 19.
        Index.clear_history(uri=index_uri, timestamp=19)

        with tiledb.Group(index_uri, "r") as group:
            assert metadata_to_list(group, "ingestion_timestamps") == [21]
            assert metadata_to_list(group, "base_sizes") == [3]
            assert group.meta["has_updates"] == 1
            if index_type != "VAMANA":
                assert metadata_to_list(group, "partition_history") == [1]
            if index_type == "VAMANA":
                assert metadata_to_list(group, "num_edges_history") == [
                    second_num_edges
                ]

        # If we load the index from timestamp 0 -> < 19, we only are returned the first two vectors.
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=10),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=19),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(0, 19)),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(None, 19)),
            queries=data,
            expected_result_d=default_result_d,
            expected_result_i=default_result_i,
        )

        # But if we load with timestamp > 20 then we get results for all three vectors.
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=None),
            queries=data,
            expected_result_d=[[0], [0], [0]],
            expected_result_i=[[0], [1], [2]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=1000),
            queries=data,
            expected_result_d=[[0], [0], [0]],
            expected_result_i=[[0], [1], [2]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(0, 1000)),
            queries=data,
            expected_result_d=[[0], [0], [0]],
            expected_result_i=[[0], [1], [2]],
        )
        query_and_check_equals(
            index=index_class(uri=index_uri, timestamp=(None, 21)),
            queries=data,
            expected_result_d=[[0], [0], [0]],
            expected_result_i=[[0], [1], [2]],
        )


def test_ingestion_with_updates(tmp_path):
    vfs = tiledb.VFS()

    dataset_dir = os.path.join(tmp_path, "dataset")
    k = 10
    size = 1000
    partitions = 10
    dimensions = 49
    nqueries = 100
    nprobe = partitions
    num_subspaces = dimensions
    data = create_random_dataset_u8(
        nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir
    )
    dtype = np.uint8

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=os.path.join(dataset_dir, "data.u8bin"),
            partitions=partitions,
            num_subspaces=num_subspaces,
        )

        ingestion_timestamps, base_sizes = load_metadata(index_uri)
        assert base_sizes == [1000]
        assert len(ingestion_timestamps) == 1
        timestamp_5_minutes_from_now = int((time.time() + 5 * 60) * 1000)
        timestamp_5_minutes_ago = int((time.time() - 5 * 60) * 1000)
        assert (
            ingestion_timestamps[0] > timestamp_5_minutes_ago
            and ingestion_timestamps[0] < timestamp_5_minutes_from_now
        )
        ingestion_timestamp = ingestion_timestamps[0]

        _, result = index.query(queries, k=k, nprobe=nprobe)
        if index_type == "IVF_PQ":
            # TODO(paris): We get 0.989 accuracy instead of 1.0. Investigate why - it should be 1.0
            # when we have `nprobe = partitions` and `num_subspaces = dimensions`.
            assert accuracy(result, gt_i) > 0.9
            continue
        assert accuracy(result, gt_i) == 1.0

        update_ids_offset = MAX_UINT64 - size
        updated_ids = {}
        for i in range(100):
            index.delete(external_id=i)
            index.update(
                vector=data[i].astype(dtype), external_id=i + update_ids_offset
            )
            updated_ids[i] = i + update_ids_offset

        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0

        index = index.consolidate_updates(retrain_index=True, partitions=20)
        _, result = index.query(queries, k=k, nprobe=20)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0

        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri)
        _, result = index.query(queries, k=k, nprobe=20)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0

        ingestion_timestamps, base_sizes = load_metadata(index_uri)
        assert base_sizes == [1000, 1000]
        assert len(ingestion_timestamps) == 2
        assert ingestion_timestamps[0] == ingestion_timestamp
        assert (
            ingestion_timestamps[1] != ingestion_timestamp
            and ingestion_timestamps[1] > timestamp_5_minutes_ago
            and ingestion_timestamps[1] < timestamp_5_minutes_from_now
        )

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ingestion_with_batch_updates(tmp_path):
    vfs = tiledb.VFS()

    dataset_dir = os.path.join(tmp_path, "dataset")
    k = 10
    size = 10000
    partitions = 100
    dimensions = 100
    nqueries = 100
    nprobe = 100
    num_subspaces = 25
    data = create_random_dataset_u8(
        nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir
    )
    dtype = np.uint8

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        minimum_accuracy = 0.85 if index_type == "IVF_PQ" else 0.99

        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=os.path.join(dataset_dir, "data.u8bin"),
            partitions=partitions,
            input_vectors_per_work_item=int(size / 10),
            num_subspaces=num_subspaces,
        )
        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i) > minimum_accuracy

        update_ids = {}
        updated_ids = {}
        update_ids_offset = MAX_UINT64 - size
        for i in range(0, 1000, 2):
            updated_ids[i] = i + update_ids_offset
            update_ids[i + update_ids_offset] = i
        external_ids = np.zeros((len(updated_ids) * 2), dtype=np.uint64)
        updates = np.empty((len(updated_ids) * 2), dtype="O")
        id = 0
        for prev_id, new_id in updated_ids.items():
            external_ids[id] = prev_id
            updates[id] = np.array([], dtype=dtype)
            id += 1
            external_ids[id] = new_id
            updates[id] = data[prev_id].astype(dtype)
            id += 1

        index.update_batch(vectors=updates, external_ids=external_ids)
        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i, updated_ids=updated_ids) > minimum_accuracy

        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri)

        if index_type == "IVF_PQ":
            # TODO(SC-48888): Fix consolidation for IVF_PQ.
            continue
        index = index.consolidate_updates()
        _, result = index.query(queries, k=k, nprobe=nprobe)
        assert accuracy(result, gt_i, updated_ids=updated_ids) > minimum_accuracy

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ingestion_with_updates_and_timetravel(tmp_path):
    vfs = tiledb.VFS()

    dataset_dir = os.path.join(tmp_path, "dataset")
    k = 10
    size = 999
    partitions = 16
    dimensions = 64
    num_subspaces = dimensions
    nqueries = 85
    data = create_random_dataset_u8(
        nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir
    )
    dtype = np.uint8

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=os.path.join(dataset_dir, "data.u8bin"),
            partitions=partitions,
            index_timestamp=1,
            num_subspaces=num_subspaces,
        )

        ingestion_timestamps, base_sizes = load_metadata(index_uri)
        assert ingestion_timestamps == [1]
        assert base_sizes == [size]

        if index_type == "IVF_FLAT":
            assert index.partitions == partitions

        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i) == 1.0

        update_ids_offset = MAX_UINT64 - size
        updated_ids = {}
        timestamp_end = 102
        for i in range(2, timestamp_end):
            index.delete(external_id=i, timestamp=i)
            index.update(
                vector=data[i].astype(dtype),
                external_id=i + update_ids_offset,
                timestamp=i,
            )
            updated_ids[i] = i + update_ids_offset

        ingestion_timestamps, base_sizes = load_metadata(index_uri)
        assert ingestion_timestamps == [1]
        assert base_sizes == [size]

        index = index_class(uri=index_uri)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri, timestamp=101)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri, timestamp=(0, 101))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri, timestamp=(0, None))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri, timestamp=(2, 101))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert (
            0.05
            <= accuracy(result, gt_i, updated_ids=updated_ids, only_updated_ids=True)
            <= 0.15
        )
        index = index_class(uri=index_uri, timestamp=(2, None))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert (
            0.05
            <= accuracy(result, gt_i, updated_ids=updated_ids, only_updated_ids=True)
            <= 0.15
        )

        # Timetravel with partial read from updates table
        updated_ids_part = {}
        for i in range(2, 52):
            updated_ids_part[i] = i + update_ids_offset
        index = index_class(uri=index_uri, timestamp=51)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids_part) == 1.0
        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri, timestamp=(0, 51))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids_part) == 1.0
        index = index_class(uri=index_uri, timestamp=(2, 51))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert (
            0.02
            <= accuracy(result, gt_i, updated_ids=updated_ids, only_updated_ids=True)
            <= 0.07
        )

        # Timetravel at previous ingestion timestamp
        index = index_class(uri=index_uri, timestamp=1)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i) == 1.0

        # Consolidate updates
        if index_type == "IVF_PQ":
            # TODO(SC-48888): Fix consolidation for IVF_PQ.
            continue
        index = index.consolidate_updates()

        ingestion_timestamps, base_sizes = load_metadata(index_uri)
        assert ingestion_timestamps == [1, timestamp_end]
        assert base_sizes == [size, size]

        index = index_class(uri=index_uri)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri, timestamp=101)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri, timestamp=(0, 101))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri, timestamp=(0, None))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri, timestamp=(2, 101))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert (
            0.05
            <= accuracy(result, gt_i, updated_ids=updated_ids, only_updated_ids=True)
            <= 0.15
        )
        index = index_class(uri=index_uri, timestamp=(2, None))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert (
            0.05
            <= accuracy(result, gt_i, updated_ids=updated_ids, only_updated_ids=True)
            <= 0.15
        )

        # Timetravel with partial read from updates table
        updated_ids_part = {}
        for i in range(2, 52):
            updated_ids_part[i] = i + update_ids_offset
        index = index_class(uri=index_uri, timestamp=51)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids_part) == 1.0
        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri, timestamp=(0, 51))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids_part) == 1.0
        index = index_class(uri=index_uri, timestamp=(2, 51))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert (
            0.02
            <= accuracy(result, gt_i, updated_ids=updated_ids, only_updated_ids=True)
            <= 0.07
        )

        # Timetravel at previous ingestion timestamp
        index = index_class(uri=index_uri, timestamp=1)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i) == 1.0
        index = index_class(uri=index_uri, timestamp=(0, 1))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i) == 1.0

        with tiledb.Group(index_uri, "r") as group:
            assert metadata_to_list(group, "ingestion_timestamps") == [1, 102]
            assert metadata_to_list(group, "base_sizes") == [size, size]
            if index_type == "VAMANA":
                num_edges_history = metadata_to_list(group, "num_edges_history")
                assert len(num_edges_history) == 2
                second_num_edges = num_edges_history[1]

        # Clear history before the latest ingestion
        assert index.latest_ingestion_timestamp == 102
        Index.clear_history(
            uri=index_uri, timestamp=index.latest_ingestion_timestamp - 1
        )

        with tiledb.Group(index_uri, "r") as group:
            assert metadata_to_list(group, "ingestion_timestamps") == [102]
            assert metadata_to_list(group, "base_sizes") == [size]
            if index_type == "VAMANA":
                assert metadata_to_list(group, "num_edges_history") == [
                    second_num_edges
                ]

        index = index_class(uri=index_uri, timestamp=1)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=51)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=101)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri, timestamp=(0, 51))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri, timestamp=(0, 101))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=(0, None))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0
        index = index_class(uri=index_uri, timestamp=(2, 51))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=(2, 101))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=(2, None))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 1.0

        # Clear all history
        Index.clear_history(uri=index_uri, timestamp=index.latest_ingestion_timestamp)
        index = index_class(uri=index_uri, timestamp=1)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=51)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=101)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri)
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=(0, 51))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=(0, 101))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri, timestamp=(0, None))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=(2, 51))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=(2, 101))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0
        index = index_class(uri=index_uri, timestamp=(2, None))
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i, updated_ids=updated_ids) == 0.0

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ingestion_with_additions_and_timetravel(tmp_path):
    vfs = tiledb.VFS()

    dataset_dir = os.path.join(tmp_path, "dataset")
    k = 100
    size = 100
    partitions = 10
    dimensions = 128
    num_subspaces = dimensions / 8
    nqueries = 1
    data = create_random_dataset_u8(
        nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir
    )
    dtype = np.uint8

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)

    for index_type, index_class in zip(INDEXES, INDEX_CLASSES):
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=os.path.join(dataset_dir, "data.u8bin"),
            partitions=partitions,
            index_timestamp=1,
            num_subspaces=num_subspaces,
        )
        if index_type == "IVF_FLAT":
            assert index.partitions == partitions
        _, result = index.query(queries, k=k, nprobe=partitions)
        assert accuracy(result, gt_i) == 1.0

        update_ids_offset = MAX_UINT64 - size
        updated_ids = {}
        for i in range(100):
            index.update(
                vector=data[i].astype(dtype),
                external_id=i + update_ids_offset,
                timestamp=i + 2,
            )
            updated_ids[i] = i + update_ids_offset

        index_uri = move_local_index_to_new_location(index_uri)
        index = index_class(uri=index_uri)
        _, result = index.query(queries, k=k, nprobe=partitions, l_search=k * 2)
        assert 0.45 < accuracy(result, gt_i)

        if index_type == "IVF_PQ":
            # TODO(SC-48888): Fix consolidation for IVF_PQ.
            continue
        index = index.consolidate_updates()
        _, result = index.query(queries, k=k, nprobe=partitions, l_search=k * 2)
        assert 0.45 < accuracy(result, gt_i)

        assert vfs.dir_size(index_uri) > 0
        Index.delete_index(uri=index_uri, config={})
        assert vfs.dir_size(index_uri) == 0


def test_ivf_flat_ingestion_tdb_random_sampling_policy(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    os.mkdir(dataset_dir)
    # data.shape should give you (cols, rows). So we transpose this before using it.
    data = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
            [4.0, 4.1, 4.2, 4.3],
            [5.0, 5.1, 5.2, 5.3],
            [6.0, 6.1, 6.2, 6.3],
            [7.0, 7.1, 7.2, 7.3],
            [8.0, 8.1, 8.2, 8.3],
            [9.0, 9.1, 9.2, 9.3],
        ],
        dtype=np.float32,
    ).transpose()
    create_array(path=os.path.join(dataset_dir, "data.tdb"), data=data)

    for training_sample_size in [3, 5, 9]:
        for input_vectors_per_work_item_during_sampling in [
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            20,
            50,
        ]:
            index_uri = os.path.join(
                tmp_path,
                f"array_{training_sample_size}_{input_vectors_per_work_item_during_sampling}",
            )
            index = ingest(
                index_type="IVF_FLAT",
                index_uri=index_uri,
                source_uri=os.path.join(dataset_dir, "data.tdb"),
                training_sampling_policy=TrainingSamplingPolicy.RANDOM,
                training_sample_size=training_sample_size,
                input_vectors_per_work_item_during_sampling=input_vectors_per_work_item_during_sampling,
                use_sklearn=True,
            )

            check_training_input_vectors(
                index_uri=index_uri,
                expected_training_sample_size=training_sample_size,
                expected_dimensions=data.shape[0],
            )

            queries = np.array([data.transpose()[3]], dtype=np.float32)
            query_and_check_equals(
                index=index,
                queries=queries,
                expected_result_d=[[0]],
                expected_result_i=[[3]],
            )


def test_ivf_flat_ingestion_fvec_random_sampling_policy(tmp_path):
    source_uri = siftsmall_inputs_file
    queries_uri = siftsmall_query_file
    gt_uri = siftsmall_groundtruth_file
    index_uri = os.path.join(tmp_path, "array")
    k = 100
    partitions = 50
    nqueries = 100
    nprobe = 20

    queries = load_fvecs(queries_uri)
    gt_i, gt_d = get_groundtruth_ivec(gt_uri, k=k, nqueries=nqueries)

    training_sample_size = 1239
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=source_uri,
        partitions=partitions,
        training_sampling_policy=TrainingSamplingPolicy.RANDOM,
        training_sample_size=training_sample_size,
        input_vectors_per_work_item=1000,
    )

    check_training_input_vectors(
        index_uri=index_uri,
        expected_training_sample_size=training_sample_size,
        expected_dimensions=queries.shape[1],
    )

    _, result = index.query(queries, k=k, nprobe=nprobe)
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY


def test_storage_versions(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    k = 10
    size = 1000
    partitions = 10
    dimensions = 128
    num_subspaces = dimensions / 2
    nqueries = 100
    data = create_random_dataset_u8(
        nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir
    )
    source_uri = os.path.join(dataset_dir, "data.u8bin")

    dtype = np.uint8
    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, _ = get_groundtruth(dataset_dir, k)

    for index_type, index_class, index_file in zip(INDEXES, INDEX_CLASSES, INDEX_FILES):
        minimum_accuracy = (
            MINIMUM_ACCURACY_IVF_PQ if index_type == "IVF_PQ" else MINIMUM_ACCURACY
        )

        # TODO(paris): Fix Vamana old storage versions and re-enable.
        if is_type_erased_index(index_type):
            continue

        # First we test with an invalid storage version.
        with pytest.raises(ValueError) as error:
            index_uri = os.path.join(tmp_path, f"array_{index_type}_invalid")
            ingest(
                index_type=index_type,
                index_uri=index_uri,
                source_uri=source_uri,
                partitions=partitions,
                storage_version="Foo",
                num_subspaces=num_subspaces,
            )
        assert "Invalid storage version" in str(error.value)

        with pytest.raises(ValueError) as error:
            index_file.create(
                uri=index_uri,
                dimensions=3,
                vector_type=np.dtype(dtype),
                storage_version="Foo",
                num_subspaces=num_subspaces,
            )
        assert "Invalid storage version" in str(error.value)

        # Then we test with valid storage versions.
        for storage_version, _ in tiledb.vector_search.storage_formats.items():
            index_uri = os.path.join(tmp_path, f"array_{index_type}_{storage_version}")
            index = ingest(
                index_type=index_type,
                index_uri=index_uri,
                source_uri=source_uri,
                partitions=partitions,
                storage_version=storage_version,
                num_subspaces=num_subspaces,
            )
            _, result = index.query(queries, k=k)
            assert accuracy(result, gt_i) >= minimum_accuracy

            update_ids_offset = MAX_UINT64 - size
            updated_ids = {}
            for i in range(10):
                index.delete(external_id=i)
                index.update(
                    vector=data[i].astype(dtype), external_id=i + update_ids_offset
                )
                updated_ids[i] = i + update_ids_offset

            _, result = index.query(queries, k=k)
            assert accuracy(result, gt_i, updated_ids=updated_ids) >= minimum_accuracy

            index = index.consolidate_updates(retrain_index=True, partitions=20)
            _, result = index.query(queries, k=k)
            assert accuracy(result, gt_i, updated_ids=updated_ids) >= minimum_accuracy

            index_uri = move_local_index_to_new_location(index_uri)
            index_ram = index_class(uri=index_uri)
            _, result = index_ram.query(queries, k=k)
            assert accuracy(result, gt_i) > minimum_accuracy


def test_ivf_flat_copy_centroids_uri(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    os.mkdir(dataset_dir)

    # Create the index data.
    data = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 3, 3]],
        dtype=np.float32,
    )

    # Create the centroids - this is based on ivf_flat_index.py.
    centroids = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
    centroids_in_size = centroids.shape[0]
    dimensions = centroids.shape[1]
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            *[
                tiledb.Dim(
                    name="rows",
                    domain=(0, dimensions - 1),
                    tile=dimensions,
                    dtype=np.dtype(np.int32),
                ),
                tiledb.Dim(
                    name="cols",
                    domain=(0, np.iinfo(np.dtype("int32")).max),
                    tile=100000,
                    dtype=np.dtype(np.int32),
                ),
            ]
        ),
        sparse=False,
        attrs=[
            tiledb.Attr(
                name="centroids",
                dtype="float32",
                filters=tiledb.FilterList([tiledb.ZstdFilter()]),
            )
        ],
        cell_order="col-major",
        tile_order="col-major",
    )
    centroids_uri = os.path.join(dataset_dir, "centroids.tdb")
    tiledb.Array.create(centroids_uri, schema)
    index_timestamp = int(time.time() * 1000)
    with tiledb.open(centroids_uri, mode="w", timestamp=index_timestamp) as A:
        A[0:dimensions, 0:centroids_in_size] = centroids.transpose()

    # Create the index.
    index_uri = os.path.join(tmp_path, "array")
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=data,
        copy_centroids_uri=centroids_uri,
        partitions=centroids_in_size,
    )

    # Query the index.
    queries = np.array([data[4]], dtype=np.float32)
    query_and_check_equals(
        index=index, queries=queries, expected_result_d=[[0]], expected_result_i=[[4]]
    )


def test_kmeans():
    k = 128
    d = 16
    n = k * k
    max_iter = 16
    n_init = 10
    verbose = False

    import sklearn.model_selection
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X, _, centers = make_blobs(
        n_samples=n, n_features=d, centers=k, return_centers=True, random_state=1
    )
    X = X.astype("float32")

    data, queries = sklearn.model_selection.train_test_split(
        X, test_size=0.1, random_state=1
    )

    np.array(
        [
            [1.0573647, 5.082087],
            [-6.229642, -1.3590931],
            [0.7446737, 6.3828287],
            [-7.698864, -3.0493321],
            [2.1362762, -4.4448104],
            [1.04019, -4.0389647],
            [0.38996044, 5.7235265],
            [1.7470839, -4.717076],
        ]
    ).astype("float32")
    np.array([[-7.3712273, -1.1178735]]).astype("float32")

    km = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        init="random",
        random_state=1,
    )
    km.fit(data)
    centroids_sk = km.cluster_centers_
    results_sk = km.predict(queries)

    centroids_tdb = kmeans_fit(
        k,
        "random",
        max_iter,
        verbose,
        n_init,
        array_to_matrix(np.transpose(data)),
        seed=1,
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

    km = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        init="k-means++",
        random_state=1,
    )
    km.fit(data)
    centroids_sk = km.cluster_centers_
    results_sk = km.predict(queries)

    assert tdb_score < 1.5 * sklearn_score

    centroids_tdb = kmeans_fit(
        k,
        "k-means++",
        max_iter,
        verbose,
        n_init,
        array_to_matrix(np.transpose(data)),
        seed=1,
    )
    centroids_tdb_np = np.transpose(np.array(centroids_tdb))
    results_tdb = kmeans_predict(centroids_tdb, array_to_matrix(np.transpose(queries)))
    results_tdb_np = np.transpose(np.array(results_tdb))

    sklearn_score = get_score(centroids_sk, results_sk)
    tdb_score = get_score(centroids_tdb_np, results_tdb_np)

    assert tdb_score < 1.5 * sklearn_score


def test_ivf_flat_ingestion_with_training_source_uri_f32(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    data = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
            [4.0, 4.1, 4.2, 4.3],
            [5.0, 5.1, 5.2, 5.3],
        ],
        dtype=np.float32,
    )
    create_manual_dataset_f32_only_data(data=data, path=dataset_dir)

    training_data = np.array([data[0], data[1], data[2]], dtype=np.float32)
    create_manual_dataset_f32_only_data(
        data=training_data, path=dataset_dir, dataset_name="training_data.f32bin"
    )
    index_uri = os.path.join(tmp_path, "array")
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.f32bin"),
        training_source_uri=os.path.join(dataset_dir, "training_data.f32bin"),
    )

    queries = np.array([data[1]], dtype=np.float32)
    query_and_check_equals(
        index=index, queries=queries, expected_result_d=[[0]], expected_result_i=[[1]]
    )

    index_uri = move_local_index_to_new_location(index_uri)
    index = IVFFlatIndex(uri=index_uri)
    query_and_check_equals(
        index=index, queries=queries, expected_result_d=[[0]], expected_result_i=[[1]]
    )

    # Also test that we can ingest with training_source_type.
    ingest(
        index_type="IVF_FLAT",
        index_uri=os.path.join(tmp_path, "array_2"),
        source_uri=os.path.join(dataset_dir, "data.f32bin"),
        training_source_uri=os.path.join(dataset_dir, "training_data.f32bin"),
        training_source_type="F32BIN",
    )


def test_ivf_flat_ingestion_with_training_source_uri_tdb(tmp_path):
    ################################################################################################
    # First set up the data.
    ################################################################################################
    dataset_dir = os.path.join(tmp_path, "dataset")
    os.mkdir(dataset_dir)
    # data.shape should give you (cols, rows). So we transpose this before using it.
    data = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
            [4.0, 4.1, 4.2, 4.3],
            [5.0, 5.1, 5.2, 5.3],
        ],
        dtype=np.float32,
    ).transpose()
    create_array(path=os.path.join(dataset_dir, "data.tdb"), data=data)

    training_data = np.array(
        [[1.0, 1.1, 1.2, 1.3], [5.0, 5.1, 5.2, 5.3]], dtype=np.float32
    ).transpose()
    create_array(
        path=os.path.join(dataset_dir, "training_data.tdb"), data=training_data
    )

    # Run a quick test that if we set up training_data incorrectly, we will raise an exception.
    with pytest.raises(ValueError) as error:
        training_data_invalid = np.array(
            [[1.0, 1.1, 1.2], [5.0, 5.1, 5.2]], dtype=np.float32
        ).transpose()
        create_array(
            path=os.path.join(dataset_dir, "training_data_invalid.tdb"),
            data=training_data_invalid,
        )
        index = ingest(
            index_type="IVF_FLAT",
            index_uri=os.path.join(tmp_path, "array_invalid"),
            source_uri=os.path.join(dataset_dir, "data.tdb"),
            training_source_uri=os.path.join(dataset_dir, "training_data_invalid.tdb"),
        )
    assert "training data dimensions" in str(error.value)

    ################################################################################################
    # Test we can ingest, query, update, and consolidate.
    ################################################################################################
    index_uri = os.path.join(tmp_path, "array")
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.tdb"),
        training_source_uri=os.path.join(dataset_dir, "training_data.tdb"),
    )

    queries = np.array([data.transpose()[1]], dtype=np.float32)
    query_and_check_equals(
        index=index, queries=queries, expected_result_d=[[0]], expected_result_i=[[1]]
    )

    update_vectors = np.empty([3], dtype=object)
    update_vectors[0] = np.array([6.0, 6.1, 6.2, 6.3], dtype=np.dtype(np.float32))
    update_vectors[1] = np.array([7.0, 7.1, 7.2, 7.3], dtype=np.dtype(np.float32))
    update_vectors[2] = np.array([8.0, 8.1, 8.2, 8.3], dtype=np.dtype(np.float32))
    index.update_batch(
        vectors=update_vectors, external_ids=np.array([1000, 1001, 1002])
    )

    index = index.consolidate_updates()

    queries = np.array([update_vectors[2]], dtype=np.float32)
    query_and_check_equals(
        index=index,
        queries=queries,
        expected_result_d=[[0]],
        expected_result_i=[[1002]],
    )

    ################################################################################################
    # Test we can load the index again and query, update, and consolidate.
    ################################################################################################
    index_uri = move_local_index_to_new_location(index_uri)

    # Load the index again and query.
    index = IVFFlatIndex(uri=index_uri)

    query_and_check_equals(
        index=index,
        queries=queries,
        expected_result_d=[[0]],
        expected_result_i=[[1002]],
    )

    # Update the index and query.
    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([9.0, 9.1, 9.2, 9.3], dtype=np.dtype(np.float32))
    update_vectors[1] = np.array([10.0, 10.1, 10.2, 10.3], dtype=np.dtype(np.float32))
    index.update_batch(vectors=update_vectors, external_ids=np.array([1003, 1004]))
    index = index.consolidate_updates()

    queries = np.array([update_vectors[0]], dtype=np.float32)
    query_and_check_equals(
        index=index,
        queries=queries,
        expected_result_d=[[0]],
        expected_result_i=[[1003]],
    )

    # Clear the index history, load, update, and query.
    Index.clear_history(uri=index_uri, timestamp=index.latest_ingestion_timestamp - 1)

    index = IVFFlatIndex(uri=index_uri)

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([11.0, 11.1, 11.2, 11.3], dtype=np.dtype(np.float32))
    update_vectors[1] = np.array([12.0, 12.1, 12.2, 12.3], dtype=np.dtype(np.float32))
    index.update_batch(vectors=update_vectors, external_ids=np.array([1003, 1004]))
    index = index.consolidate_updates()

    queries = np.array([update_vectors[0]], dtype=np.float32)
    query_and_check_equals(
        index=index,
        queries=queries,
        expected_result_d=[[0]],
        expected_result_i=[[1003]],
    )

    ###############################################################################################
    # Also test that we can ingest with training_source_type.
    ###############################################################################################
    ingest(
        index_type="IVF_FLAT",
        index_uri=os.path.join(tmp_path, "array_2"),
        source_uri=os.path.join(dataset_dir, "data.tdb"),
        training_source_uri=os.path.join(dataset_dir, "training_data.tdb"),
        training_source_type="TILEDB_ARRAY",
    )


def test_ivf_flat_ingestion_with_training_source_uri_numpy(tmp_path):
    ################################################################################################
    # First set up the data.
    ################################################################################################
    data = np.array(
        [
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
            [4.0, 4.1, 4.2, 4.3],
            [5.0, 5.1, 5.2, 5.3],
        ],
        dtype=np.float32,
    )
    training_data = data[1:3]

    # Run a quick test that if we set up training_data incorrectly, we will raise an exception.
    with pytest.raises(ValueError) as error:
        training_data_invalid = np.array(
            [[4.0, 4.1, 4.2], [5.0, 5.1, 5.2]], dtype=np.float32
        )
        index = ingest(
            index_type="IVF_FLAT",
            index_uri=os.path.join(tmp_path, "array_invalid"),
            input_vectors=data,
            training_input_vectors=training_data_invalid,
        )
    assert "training data dimensions" in str(error.value)

    ################################################################################################
    # Test we can ingest, query, update, and consolidate.
    ################################################################################################
    index_uri = os.path.join(tmp_path, "array")
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        input_vectors=data,
        training_input_vectors=training_data,
    )

    queries = np.array([data[1]], dtype=np.float32)
    query_and_check_equals(
        index=index, queries=queries, expected_result_d=[[0]], expected_result_i=[[1]]
    )

    update_vectors = np.empty([3], dtype=object)
    update_vectors[0] = np.array([6.0, 6.1, 6.2, 6.3], dtype=np.dtype(np.float32))
    update_vectors[1] = np.array([7.0, 7.1, 7.2, 7.3], dtype=np.dtype(np.float32))
    update_vectors[2] = np.array([8.0, 8.1, 8.2, 8.3], dtype=np.dtype(np.float32))
    index.update_batch(
        vectors=update_vectors, external_ids=np.array([1000, 1001, 1002])
    )

    index = index.consolidate_updates()

    queries = np.array([update_vectors[2]], dtype=np.float32)
    query_and_check_equals(
        index=index,
        queries=queries,
        expected_result_d=[[0]],
        expected_result_i=[[1002]],
    )

    ################################################################################################
    # Test we can load the index again and query, update, and consolidate.
    ################################################################################################
    index_uri = move_local_index_to_new_location(index_uri)
    index = IVFFlatIndex(uri=index_uri)

    queries = np.array([data[1]], dtype=np.float32)
    query_and_check_equals(
        index=index, queries=queries, expected_result_d=[[0]], expected_result_i=[[1]]
    )

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([9.0, 9.1, 9.2, 9.3], dtype=np.dtype(np.float32))
    update_vectors[1] = np.array([10.0, 10.1, 10.2, 10.3], dtype=np.dtype(np.float32))
    index.update_batch(vectors=update_vectors, external_ids=np.array([1003, 1004]))
    index = index.consolidate_updates()

    queries = np.array([update_vectors[0]], dtype=np.float32)
    query_and_check_equals(
        index=index,
        queries=queries,
        expected_result_d=[[0]],
        expected_result_i=[[1003]],
    )

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([11.0, 11.1, 11.2, 11.3], dtype=np.dtype(np.float32))
    update_vectors[1] = np.array([12.0, 12.1, 12.2, 12.3], dtype=np.dtype(np.float32))
    index.update_batch(vectors=update_vectors, external_ids=np.array([1003, 1004]))
    index = index.consolidate_updates(retrain_index=True, training_sample_size=3)

    queries = np.array([update_vectors[0]], dtype=np.float32)
    query_and_check_equals(
        index=index,
        queries=queries,
        expected_result_d=[[0]],
        expected_result_i=[[1003]],
    )


def test_ivf_flat_taskgraph_query(tmp_path):
    dataset_dir = os.path.join(tmp_path, "dataset")
    index_uri = os.path.join(tmp_path, "array")
    k = 10
    size = 10000
    partitions = 100
    dimensions = 129
    nqueries = 100
    nprobe = 20
    create_random_dataset_u8(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    dtype = np.uint8

    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, gt_d = get_groundtruth(dataset_dir, k)
    index = ingest(
        index_type="IVF_FLAT",
        index_uri=index_uri,
        source_uri=os.path.join(dataset_dir, "data.u8bin"),
        partitions=partitions,
        input_vectors_per_work_item=int(size / 10),
    )
    _, result = index._taskgraph_query(
        queries, k=k, nprobe=nprobe, nthreads=8, mode=Mode.LOCAL, num_partitions=10
    )
    assert accuracy(result, gt_i) > MINIMUM_ACCURACY
