import numpy as np
from common import *

import tiledb
import tiledb.vector_search as vs
from tiledb.vector_search import _tiledbvspy as vspy

import pytest

def test_load_matrix(tmpdir):
    p = str(tmpdir.mkdir("test").join("test.tdb"))
    data = np.random.rand(12).astype(np.float32).reshape(3, 4)

    # write some test data with tiledb-py
    create_array(p, data)

    # load the data with the vector search API and compare
    m, orig_matrix = vs.load_as_array(p, return_matrix=True)
    assert np.array_equal(m, data)

    # mutate and compare again - should match backing data in the C++ Matrix
    data[0, 0] = np.random.rand(1).astype(np.float32)
    m[0, 0] = data[0, 0]
    assert np.array_equal(m, data)
    assert np.array_equal(orig_matrix[0, 0], data[0, 0])


@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/work/proj/vector-search/datasets/sift-andrew/")),
    reason="requires sift dataset")
def test_flat_query():
    # db_uri = "s3://tiledb-andrew/sift/sift_base"
    # probe_uri = "s3://tiledb-andrew/sift/sift_query"
    # g_uri = "s3://tiledb-andrew/sift/sift_groundtruth"

    db_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_base"
    probe_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_query"
    g_uri = (
        "/Users/inorton/work/proj/vector-search/datasets/sift-andrew/sift_groundtruth"
    )

    k = 10
    nqueries = 1

    db = vs.load_as_matrix(db_uri)
    targets = vs.load_as_matrix(probe_uri, nqueries)  # TODO: make 2nd optional

    r = vs.query_vq(db, targets, k, nqueries, 8)  # k  # nqueries  # nthreads

    ra = np.array(r, copy=False)
    print(ra)
    print(ra.shape)

    g_array = tiledb.open(g_uri)
    g = g_array[:]["a"]

    # validate top_k
    assert np.array_equal(np.sort(ra[:k], axis=0), np.sort(g[:k, :nqueries], axis=0))

    g_m = vs.load_as_matrix(g_uri)
    assert vspy.validate_top_k_u64(r, g_m)
