import numpy as np
from common import *
import pytest

from tiledb.vector_search.flat_index import FlatIndex
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
from tiledb.vector_search.utils import load_fvecs

MINIMUM_ACCURACY = 0.85

def test_create_and_query_indices_with_old_storage_versions(tmp_path):
    '''
    Tests that the current code can create indices using older storage version formats and then 
    query them.
    '''
    dataset_dir = os.path.join(tmp_path, "dataset")
    k = 10
    size = 1000
    partitions = 10
    dimensions = 128
    nqueries = 100
    data = create_random_dataset_u8(nb=size, d=dimensions, nq=nqueries, k=k, path=dataset_dir)
    source_uri = os.path.join(dataset_dir, "data.u8bin")

    dtype = np.uint8
    queries = get_queries(dataset_dir, dtype=dtype)
    gt_i, _ = get_groundtruth(dataset_dir, k)
    
    indexes = ["FLAT", "IVF_FLAT"]
    index_classes = [FlatIndex, IVFFlatIndex]
    index_files = [tiledb.vector_search.flat_index, tiledb.vector_search.ivf_flat_index]
    for index_type, index_class, index_file in zip(indexes, index_classes, index_files):
        # First we test with an invalid storage version.
        with pytest.raises(ValueError) as error:
            index_uri = os.path.join(tmp_path, f"array_{index_type}_invalid")
            ingest(
                index_type=index_type,
                index_uri=index_uri,
                source_uri=source_uri,
                partitions=partitions,
                storage_version="Foo"
            )
        assert "Invalid storage version" in str(error.value)

        with pytest.raises(ValueError) as error:
            index_file.create(uri=index_uri, dimensions=3, vector_type=np.dtype(dtype), storage_version="Foo")
        assert "Invalid storage version" in str(error.value)

        # Then we test with valid storage versions.
        for storage_version, _ in tiledb.vector_search.storage_formats.items():
            index_uri = os.path.join(tmp_path, f"array_{index_type}_{storage_version}")
            index = ingest(
                index_type=index_type,
                index_uri=index_uri,
                source_uri=source_uri,
                partitions=partitions,
                storage_version=storage_version
            )
            _, result = index.query(queries, k=k)
            assert accuracy(result, gt_i) >= MINIMUM_ACCURACY

            update_ids_offset = MAX_UINT64 - size
            updated_ids = {}
            for i in range(10):
                index.delete(external_id=i)
                index.update(vector=data[i].astype(dtype), external_id=i + update_ids_offset)
                updated_ids[i] = i + update_ids_offset

            _, result = index.query(queries, k=k)
            assert accuracy(result, gt_i, updated_ids=updated_ids) >= MINIMUM_ACCURACY

            index = index.consolidate_updates(retrain_index=True, partitions=20)
            _, result = index.query(queries, k=k)
            assert accuracy(result, gt_i, updated_ids=updated_ids) >= MINIMUM_ACCURACY

            index_ram = index_class(uri=index_uri)
            _, result = index_ram.query(queries, k=k)
            assert accuracy(result, gt_i) > MINIMUM_ACCURACY

def test_query_old_indices():
    '''
    Tests that current code can query indices which were written to disk by old code.
    '''
    backwards_compatibility_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backwards-compatibility-data')
    datasets_path = os.path.join(backwards_compatibility_path, 'data')
    base = load_fvecs(os.path.join(backwards_compatibility_path, 'siftmicro_base.fvecs'))
    query_indices = [0, 3, 4, 8, 10, 19, 28, 31, 39, 40, 41, 47, 49, 50, 56, 64, 68, 70, 71, 79, 82, 89, 90, 94]
    queries = base[query_indices]

    for directory_name in os.listdir(datasets_path):
        version_path = os.path.join(datasets_path, directory_name)
        if not os.path.isdir(version_path):
            continue

        for index_name in os.listdir(version_path):
            index_uri = os.path.join(version_path, index_name)
            if not os.path.isdir(index_uri):
              continue

            if "ivf_flat" in index_name:
                index = IVFFlatIndex(uri=index_uri)
            elif "flat" in index_name:
                index = FlatIndex(uri=index_uri)
            else:
                assert False, f"Unknown index name: {index_name}"

            result_d, result_i = index.query(queries, k=1)
            assert query_indices == result_i.flatten().tolist()
            assert result_d.flatten().tolist() == [0 for _ in range(len(query_indices))]