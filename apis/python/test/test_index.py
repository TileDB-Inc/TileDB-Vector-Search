import tiledb
import tiledb.vector_search as vs
from tiledb.vector_search.index import FlatIndex

import numpy as np

def test_flat_index(tmpdir):
    #db_uri = "s3://tiledb-andrew/sift/sift_base"
    #ground_truth_uri = "s3://tiledb-andrew/sift/sift_groundtruth"

    db_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_base"
    ground_truth_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_groundtruth"
    query_uri = "~/work/proj/vector-search/datasets/sift-andrew/sift_query"

    query_vectors = vs.load_as_array(query_uri)[:,:10]

    index = FlatIndex(db_uri)
    result = index.query(query_vectors)
    assert isinstance(result, np.ndarray)

    #ground_truth = vs.load_as_array(ground_truth_uri)

def test_kmeans_index(tmpdir):
    p = str(tmpdir.mkdir("test").join("test.tdb"))
    base = "~/work/proj/vector-search/datasets/sift-andrew/"

    #db_uri = f"{base}/sift_base"
    #groundtruth_uri = f"{base}/sift_groundtruth"

    parts_db_uri = f"{base}/parts.tdb"
    centroids_uri = f"{base}/centroids.tdb"

    query_uri = f"{base}/sift_query"
    index_uri = f"{base}/index.tdb"
    ids_uri = f"{base}/ids.tdb"

    index = vs.KMeansIndex(
        parts_db_uri,
        centroids_uri,
        index_uri,
        ids_uri
    )

    query_vectors = vs.load_as_matrix(query_uri, 10)
    query_array = np.array(query_vectors, copy=False)

    r = index.query(query_array)
    print(r)