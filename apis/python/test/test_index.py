import tiledb
import tiledb.vector_search as vs
from tiledb.vector_search.index import FlatIndex

import numpy as np


def test_flat_index(tmpdir):
    group_uri = "~/work/proj/vector-search/datasets/sift-andrew/"
    ground_truth_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_groundtruth"
    query_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_query"

    query_vectors = vs.load_as_array(query_uri)[:, :10]

    index = FlatIndex(group_uri, parts_name="sift_base")
    result = index.query(query_vectors)
    assert isinstance(result, np.ndarray)

    # ground_truth = vs.load_as_array(ground_truth_uri)