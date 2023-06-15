import tiledb
import tiledb.vector_search as vs
from tiledb.vector_search.index import FlatIndex

import numpy as np


def test_flat_index(tmpdir):
    # db_uri = "s3://tiledb-andrew/sift/sift_base"
    # ground_truth_uri = "s3://tiledb-andrew/sift/sift_groundtruth"

    db_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_base"
    ground_truth_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_groundtruth"
    query_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_query"

    query_vectors = vs.load_as_array(query_uri)[:, :10]

    index = FlatIndex(db_uri)
    result = index.query(query_vectors)
    assert isinstance(result, np.ndarray)

    # ground_truth = vs.load_as_array(ground_truth_uri)
