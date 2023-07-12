import numpy as np

import tiledb.vector_search as vs
from tiledb.vector_search.index import FlatIndex

import numpy as np
import os
import pytest


## only run this test if the dataset is available
@pytest.mark.skipif(
    not os.path.exists(
        os.path.expanduser("~/work/proj/vector-search/datasets/sift-andrew/")
    ),
    reason="requires sift dataset",
)
def test_flat_index_local():
    group_uri = "~/work/proj/vector-search/datasets/sift-andrew/"
    query_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_query"
    ground_truth_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_groundtruth"

    nquery = 10
    query_vectors = vs.load_as_array(query_uri)[:, :nquery]

    index = FlatIndex(group_uri, dtype="float32", parts_name="sift_base")
    result = index.query(query_vectors)
    assert isinstance(result, np.ndarray)

    ground_truth = vs.load_as_matrix(ground_truth_uri, nquery)
    result_m = vs.array_to_matrix(result)
    vs.validate_top_k(result_m, ground_truth)
