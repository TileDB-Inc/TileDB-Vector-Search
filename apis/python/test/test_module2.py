
import tiledb
import tiledb.vector_search as vs
from tiledb.vector_search import _tiledbvspy as vspy

import numpy as np
import logging



m1_root = "/Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/"
db_uri = m1_root + "sift/sift_base"
centroids_uri=    m1_root + "sift/centroids"
parts_uri=    m1_root + "sift/parts"
index_uri=    m1_root + "sift/index"
sizes_uri=    m1_root + "/1M/index_size.tdb"
ids_uri=    m1_root + "sift/ids"
query_uri=    m1_root + "sift/sift_query"
groundtruth_uri=    m1_root + "sift/sift_groundtruth"

bigann1M_base_uri=m1_root + "1M/bigann1M_base"
bigann1M_query_uri=m1_root + "1M/query_public_10k"
bigann1M_groundtruth_uri=m1_root + "1M/bigann_1M_GT_nnids"
bigann1M_centroids_uri=m1_root + "1M/centroids.tdb"
bigann1M_ids_uri=m1_root + "1M/ids.tdb"
bigann1M_index_uri=m1_root + "1M/index.tdb"
bigann1M_index_size_uri=m1_root + "1M/index_size.tdb"
bigann1M_parts_uri=m1_root + "1M/parts.tdb"
bigann1M_flatIVF_index_uri_64_32=    m1_root + "1M/flatIVF_index_1M_base_64_32"

fmnist_train_uri=m1_root + "fmnist/fmnist_train.tdb"
fmnist_test_uri=m1_root + "fmnist/fmnist_test.tdb"
fmnist_groundtruth_uri=    m1_root + "fmnist/fmnist_neighbors.tdb"
sift_base_uri=m1_root + "sift/sift_base"

fmnist_distances=    m1_root + "fmnist/fmnist_distances.tdb"
fmnist_neighbors=    m1_root + "fmnist/fmnist_neighbors.tdb"
fmnist_test=m1_root + "fmnist/fmnist_test.tdb"
fmnist_train=m1_root + "fmnist/fmnist_train.tdb"

diskann_test_256bin=    m1_root + "diskann/siftsmall_learn_256pts.fbin"

siftsmall_base_uri=m1_root + "siftsmall/siftsmall_base"
siftsmall_groundtruth_uri=    m1_root + "siftsmall/siftsmall_groundtruth"
siftsmall_query_uri=m1_root + "siftsmall/siftsmall_query"
siftsmall_flatIVF_index_uri=    m1_root + "siftsmall/flatIVF_index_siftsmall_base"
siftsmall_flatIVF_index_uri_32_64=    m1_root + "siftsmall/flatIVF_index_siftsmall_base_32_64"


# ctx = tiledb.Ctx()
ctx = vspy.Ctx({})

def test_construct_FeatureVector():
    p = 0
    
def test_construct_FeatureVectorArray():
    a = vspy.FeatureVectorArray(ctx, siftsmall_base_uri)
    assert a.feature_type_string() == "float32"
    a = vspy.FeatureVectorArray(ctx, bigann1M_base_uri)
    assert a.feature_type_string() == "uint8"

def test_feature_vector_array_to_numpy():
    a = vspy.FeatureVectorArray(ctx, siftsmall_base_uri)
    assert a.num_vectors() == 10000
    assert a.dimension() == 128
    b = np.array(a)
    assert b.shape == (10000, 128)

    a = vspy.FeatureVectorArray(ctx, bigann1M_base_uri)
    assert a.num_vectors() == 1000000
    assert a.dimension() == 128
    b = np.array(a)
    assert b.shape == (1000000, 128)

def test_numpy_to_feature_vector_array():

    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (10000, 128)
    assert b.dimension() == 128
    assert b.num_vectors() == 10000

    a = np.array(np.random.rand(128, 10000), dtype=np.float32, order='F')
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (128, 10000)
    assert b.dimension() == 128
    assert b.num_vectors() == 10000

    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a.T)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (10000, 128)
    assert b.dimension() == 128
    assert b.num_vectors() == 10000

    a = np.array(np.random.rand(1000000, 128), dtype=np.uint8)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))
    assert a.shape == (1000000, 128)
    assert b.dimension() == 128
    assert b.num_vectors() == 1000000


    a = np.array(np.random.rand(10000, 128), dtype=np.float32)
    b = vspy.FeatureVectorArray(a)
    logging.info(a.shape)
    logging.info((b.dimension(), b.num_vectors()))

    c = np.array(b)
    logging.info(c.shape)

    assert a.shape == c.shape
    assert (a == c).all()

#    logging.info(a[1:5, 1:5])
#    logging.info(c[1:5, 1:5])

def test_IndexFlatL2():
    p = 0

def test_construct_IndexIVFFlat():
    a = vspy.IndexIVFFlat(ctx, siftsmall_flatIVF_index_uri_32_64)
    assert a.feature_type_string() == "float32"
    assert a.id_type_string() == "uint32"
    assert a.px_type_string() == "uint64"
    assert a.dimension() == 128

    a = vspy.IndexIVFFlat(ctx, bigann1M_flatIVF_index_uri_64_32)
    assert a.feature_type_string() == "uint8"
    assert a.id_type_string() == "uint64"
    assert a.px_type_string() == "uint32"
    assert a.dimension() == 128

