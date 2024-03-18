import numpy as np
import pytest
from array_paths import *
from common import *
import json

import tiledb.vector_search.index as ind
from tiledb.vector_search import Index
from tiledb.vector_search import flat_index
from tiledb.vector_search import ivf_flat_index
from tiledb.vector_search import vamana_index
from tiledb.vector_search.index import create_metadata
from tiledb.vector_search.index import DATASET_TYPE
from tiledb.vector_search.flat_index import FlatIndex
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
from tiledb.vector_search.vamana_index import VamanaIndex
from tiledb.vector_search.utils import load_fvecs


def query_and_check(index, queries, k, expected, **kwargs):
    for _ in range(3):
        result_d, result_i = index.query(queries, k=k, **kwargs)
        assert expected.issubset(set(result_i[0]))

def check_default_metadata(uri, expected_vector_type, expected_storage_version, expected_index_type):
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

    assert "has_updates" in group.meta
    assert group.meta["has_updates"] == False
    assert type(group.meta["has_updates"]) == np.int64

def test_flat_index(tmp_path):
    uri = os.path.join(tmp_path, "array")
    vector_type = np.dtype(np.uint8)
    index = flat_index.create(uri=uri, dimensions=3, vector_type=vector_type)
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {ind.MAX_UINT64})
    check_default_metadata(uri, vector_type, STORAGE_VERSION, "FLAT")

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([0, 0, 0], dtype=vector_type)
    update_vectors[1] = np.array([1, 1, 1], dtype=vector_type)
    update_vectors[2] = np.array([2, 2, 2], dtype=vector_type)
    update_vectors[3] = np.array([3, 3, 3], dtype=vector_type)
    update_vectors[4] = np.array([4, 4, 4], dtype=vector_type)
    index.update_batch(vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3})

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3})

    index.delete_batch(external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4})

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4})

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([1, 1, 1], dtype=vector_type)
    update_vectors[1] = np.array([3, 3, 3], dtype=vector_type)
    index.update_batch(vectors=update_vectors, external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3})

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3})

    index.delete_batch(external_ids=np.array([1, 3]))
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4})

    index = index.consolidate_updates()
    query_and_check(index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4})


def test_ivf_flat_index(tmp_path):
    partitions = 10
    uri = os.path.join(tmp_path, "array")
    vector_type = np.dtype(np.uint8)
    index = ivf_flat_index.create(
        uri=uri, dimensions=3, vector_type=vector_type, partitions=partitions
    )
    query_and_check(
        index,
        np.array([[2, 2, 2]], dtype=np.float32),
        3,
        {ind.MAX_UINT64},
        nprobe=partitions,
    )
    check_default_metadata(uri, vector_type, STORAGE_VERSION, "IVF_FLAT")

    update_vectors = np.empty([5], dtype=object)
    update_vectors[0] = np.array([0, 0, 0], dtype=vector_type)
    update_vectors[1] = np.array([1, 1, 1], dtype=vector_type)
    update_vectors[2] = np.array([2, 2, 2], dtype=vector_type)
    update_vectors[3] = np.array([3, 3, 3], dtype=vector_type)
    update_vectors[4] = np.array([4, 4, 4], dtype=vector_type)
    index.update_batch(vectors=update_vectors, external_ids=np.array([0, 1, 2, 3, 4]))

    query_and_check(
        index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3}, nprobe=partitions
    )

    index = index.consolidate_updates()

    query_and_check(
        index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3}, nprobe=partitions
    )

    index.delete_batch(external_ids=np.array([1, 3]))
    query_and_check(
        index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4}, nprobe=partitions
    )

    index = index.consolidate_updates()
    query_and_check(
        index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4}, nprobe=partitions
    )

    update_vectors = np.empty([2], dtype=object)
    update_vectors[0] = np.array([1, 1, 1], dtype=vector_type)
    update_vectors[1] = np.array([3, 3, 3], dtype=vector_type)
    index.update_batch(vectors=update_vectors, external_ids=np.array([1, 3]))
    query_and_check(
        index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3}, nprobe=partitions
    )

    index = index.consolidate_updates()
    query_and_check(
        index, np.array([[2, 2, 2]], dtype=np.float32), 3, {1, 2, 3}, nprobe=partitions
    )

    index.delete_batch(external_ids=np.array([1, 3]))
    query_and_check(
        index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4}, nprobe=partitions
    )

    index = index.consolidate_updates()
    query_and_check(
        index, np.array([[2, 2, 2]], dtype=np.float32), 3, {0, 2, 4}, nprobe=partitions
    )

def test_vamana_index(tmp_path):
    uri = os.path.join(tmp_path, "array")
    dimensions = 3
    vector_type = np.dtype(np.uint8)

    # Create the index.
    index = vamana_index.create(uri=uri, dimensions=dimensions, vector_type=vector_type, id_type=np.dtype(np.uint32))
    assert index.get_dimensions() == dimensions
    query_and_check(
        index,
        np.array([[2, 2, 2]], dtype=np.float32),
        3,
        {ind.MAX_UINT64}
    )

    # Open the index.
    index = VamanaIndex(uri=uri)
    assert index.get_dimensions() == dimensions
    query_and_check(
        index,
        np.array([[2, 2, 2]], dtype=np.float32),
        3,
        {ind.MAX_UINT64}
    )

def test_delete_invalid_index(tmp_path):
    # We don't throw with an invalid uri.
    Index.delete_index(uri="invalid_uri", config=tiledb.cloud.Config())


def test_delete_index(tmp_path):
    indexes = ["FLAT", "IVF_FLAT"]
    index_classes = [FlatIndex, IVFFlatIndex]
    data = np.array([[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]], dtype=np.float32)
    for index_type, index_class in zip(indexes, index_classes):
        index_uri = os.path.join(tmp_path, f"array_{index_type}")
        ingest(index_type=index_type, index_uri=index_uri, input_vectors=data)
        Index.delete_index(uri=index_uri, config=tiledb.cloud.Config())
        with pytest.raises(tiledb.TileDBError) as error:
            index_class(uri=index_uri)
        assert "does not exist" in str(error.value)


def test_index_with_incorrect_dimensions(tmp_path):
    indexes = [flat_index, ivf_flat_index, vamana_index]
    for index_type in indexes:
        uri = os.path.join(tmp_path, f"array_{index_type.__name__}")
        index = index_type.create(uri=uri, dimensions=3, vector_type=np.dtype(np.uint8))

        # Wrong number of dimensions will raise a TypeError.
        with pytest.raises(TypeError):
            index.query(np.array(1, dtype=np.float32), k=3)
        with pytest.raises(TypeError):
            index.query(np.array([1, 1, 1], dtype=np.float32), k=3)
        with pytest.raises(TypeError):
            index.query(np.array([[[1, 1, 1]]], dtype=np.float32), k=3)
        with pytest.raises(TypeError):
            index.query(np.array([[[[1, 1, 1]]]], dtype=np.float32), k=3)

        # Okay otherwise.
        index.query(np.array([[1, 1, 1]], dtype=np.float32), k=3)


def test_index_with_incorrect_num_of_query_columns_simple(tmp_path):
    siftsmall_uri = siftsmall_inputs_file
    queries_uri = siftsmall_query_file
    indexes = ["FLAT", "IVF_FLAT"]
    for index_type in indexes:
        index_uri = os.path.join(tmp_path, f"sift10k_flat_{index_type}")
        index = ingest(
            index_type=index_type,
            index_uri=index_uri,
            source_uri=siftsmall_uri,
            source_type="FVEC",
        )

        # Wrong number of columns will raise a TypeError.
        query_shape = (1, 1)
        with pytest.raises(TypeError):
            index.query(np.random.rand(*query_shape).astype(np.float32), k=10)

        # Okay otherwise.
        queries = load_fvecs(queries_uri)
        index.query(queries, k=10)


def test_index_with_incorrect_num_of_query_columns_complex(tmp_path):
    # Tests that we raise a TypeError if the number of columns in the query is not the same as the
    # number of columns in the indexed data.
    size = 1000
    indexes = ["FLAT", "IVF_FLAT"]
    num_columns_in_vector = [1, 2, 3, 4, 5, 10]
    for index_type in indexes:
        for num_columns in num_columns_in_vector:
            index_uri = os.path.join(tmp_path, f"array_{index_type}_{num_columns}")
            dataset_dir = os.path.join(tmp_path, f"dataset_{index_type}_{num_columns}")
            create_random_dataset_f32_only_data(
                nb=size, d=num_columns, centers=1, path=dataset_dir
            )
            index = ingest(
                index_type=index_type,
                index_uri=index_uri,
                source_uri=os.path.join(dataset_dir, "data.f32bin"),
            )

            # We have created a dataset with num_columns in each vector. Let's try creating queries
            # with different numbers of columns and confirming incorrect ones will throw.
            for num_columns_for_query in range(1, num_columns + 2):
                query_shape = (1, num_columns_for_query)
                query = np.random.rand(*query_shape).astype(np.float32)
                if query.shape[1] == num_columns:
                    index.query(query, k=1)
                else:
                    with pytest.raises(TypeError):
                        index.query(query, k=1)

                # TODO(paris): This will throw with the following error. Fix and re-enable, then remove
                # test_index_with_incorrect_num_of_query_columns_in_single_vector_query:
                #   def array_to_matrix(array: np.ndarray):
                #           if array.dtype == np.float32:
                #   >           return pyarray_copyto_matrix_f32(array)
                #   E           RuntimeError: Number of dimensions must be two
                # Here we test with a query which is just a vector, i.e. [1, 2, 3].
                # query = query[0]
                # if num_columns_for_query == num_columns:
                #     index.query(query, k=1)
                # else:
                #     with pytest.raises(TypeError):
                #         index.query(query, k=1)


def test_index_with_incorrect_num_of_query_columns_in_single_vector_query(tmp_path):
    # Tests that we raise a TypeError if the number of columns in the query is not the same as the
    # number of columns in the indexed data, specifically for a single vector query.
    # i.e. queries = [1, 2, 3]  instead of queries = [[1, 2, 3], [4, 5, 6]].
    indexes = [flat_index, ivf_flat_index]
    for index_type in indexes:
        uri = os.path.join(tmp_path, f"array_{index_type.__name__}")
        index = index_type.create(uri=uri, dimensions=3, vector_type=np.dtype(np.uint8))

        # Wrong number of columns will raise a TypeError.
        with pytest.raises(TypeError):
            index.query(np.array([1], dtype=np.float32), k=3)
        with pytest.raises(TypeError):
            index.query(np.array([1, 1], dtype=np.float32), k=3)
        with pytest.raises(TypeError):
            index.query(np.array([1, 1, 1, 1], dtype=np.float32), k=3)

        # TODO:  This also throws a TypeError for incorrect dimension
        with pytest.raises(TypeError):
            index.query(np.array([1, 1, 1], dtype=np.float32), k=3)

def test_create_metadata(tmp_path):
    uri = os.path.join(tmp_path, "array")

    # Create the metadata at the specified URI.
    dimensions = 3
    vector_type: np.dtype = np.dtype(np.uint8)
    index_type: str = "IVF_FLAT"
    storage_version: str = STORAGE_VERSION
    group_exists: bool = False
    create_metadata(uri, dimensions, vector_type, index_type, storage_version, group_exists)
    
    # Check it contains the default metadata.
    check_default_metadata(uri, vector_type, storage_version, index_type)