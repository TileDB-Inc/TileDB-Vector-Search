import numpy as np

import tiledb.vector_search as vs
from tiledb.vector_search.index import FlatIndex


def test_flat_index(tmpdir):
    group_uri = "~/work/proj/vector-search/datasets/sift-andrew/"
    query_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_query"

    query_vectors = vs.load_as_array(query_uri)[:, :10]

    index = FlatIndex(group_uri, dtype="float32", parts_name="sift_base")
    result = index.query(query_vectors)
    assert isinstance(result, np.ndarray)

    # ground_truth = vs.load_as_array(ground_truth_uri)
