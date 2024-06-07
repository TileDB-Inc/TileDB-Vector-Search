from common import *

from tiledb.vector_search.flat_index import FlatIndex
from tiledb.vector_search.ivf_flat_index import IVFFlatIndex
from tiledb.vector_search.utils import load_fvecs
from tiledb.vector_search.vamana_index import VamanaIndex

MINIMUM_ACCURACY = 0.85


def test_query_old_indices():
    """
    Tests that current code can query indices which were written to disk by old code.
    """
    backwards_compatibility_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "backwards-compatibility-data"
    )
    datasets_path = os.path.join(backwards_compatibility_path, "data")
    base = load_fvecs(
        os.path.join(backwards_compatibility_path, "siftmicro_base.fvecs")
    )
    query_indices = [
        0,
        3,
        4,
        8,
        10,
        19,
        28,
        31,
        39,
        40,
        41,
        47,
        49,
        50,
        56,
        64,
        68,
        70,
        71,
        79,
        82,
        89,
        90,
        94,
    ]
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
            elif "vamana" in index_name:
                index = VamanaIndex(uri=index_uri)
            else:
                assert False, f"Unknown index name: {index_name}"

            result_d, result_i = index.query(queries, k=1)
            assert query_indices == result_i.flatten().tolist()
            assert result_d.flatten().tolist() == [0 for _ in range(len(query_indices))]
