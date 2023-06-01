#!/bin/bash -f

ivf_hack="${HOME}/feature-vector-prototype/experimental/cmake-build-release/src/ivf_hack"

 ${ivf_hack} \
     --db_uri            's3://tiledb-andrew/sift/bigann10M_base'		                     \
     --centroids_uri     's3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/centroids.tdb'  \
     --parts_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/parts.tdb'      \
     --index_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/index.tdb'      \
     --ids_uri           's3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/ids.tdb'        \
     --nqueries           10  \
     --cluster            10  \
     -v -d \
     --query_uri         's3://tiledb-andrew/kmeans/benchmark/query_public_10k' \
     --groundtruth_uri   's3://tiledb-andrew/kmeans/benchmark/bigann_10M_GT_nnids'

# Also s3://tiledb-andrew/sift/gnd/idx_10M
# Quick spot check shows they are the same
# (Except idx_10M (et al) have 1000 rows, whereas bigann_10M has 100 rows.)

exit

${ivf_hack} \
    --db_uri            's3://tiledb-andrew/sift/bigann100M_base'		                       \
    --query_uri         's3://tiledb-andrew/kmeans/benchmark/query_public_10k' \
    --centroids_uri     's3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/centroids.tdb'  \
    --index_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/index.tdb'      \
    --parts_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/parts.tdb'      \
    --ids_uri           's3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/ids.tdb'        \
    --nqueries           0 \
    --cluster            20  \
    -v -d 

${ivf_hack} \
    --db_uri            's3://tiledb-nikos/vector-search/datasets/arrays/sift-1b-col-major' \
    --query_uri         's3://tiledb-andrew/kmeans/benchmark/query_public_10k' \
    --centroids_uri     's3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/centroids.tdb'  \
    --index_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index.tdb'      \
    --parts_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/parts.tdb'      \
    --ids_uri           's3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/ids.tdb'        \
    --nqueries          0 \
    --cluster           100 \
    -v -d 


#
#    --q_uri         's3://tiledb-andrew/kmeans/benchmark/query_public_10k' \
#    --g_uri         's3://tiledb-andrew/kmeans/benchmark/bigann_1B_GT_nnids' \

#    --groundtruth_uri   's3://tiledb-andrew/kmeans/benchmark/bigann_10M_GT_nnids' \



${ivf_hack} \
    --db_uri            's3://tiledb-andrew/sift/sift_base'                                        \
    --centroids_uri     's3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/centroids.tdb'  \
    --index_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/index.tdb'      \
    --parts_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/parts.tdb'      \
    --ids_uri           's3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/ids.tdb'        \
    --query_uri         's3://tiledb-andrew/sift/sift_query'                                       \
    --groundtruth_uri   's3://tiledb-andrew/sift/sift_groundtruth'                                 \
    --log               'ivf-shakedown.log'                                                        \
    --nqueries          0 \
    -v -d 

${ivf_hack} \
    --db_uri            's3://tiledb-andrew/sift/sift_base'                                        \
    --centroids_uri     's3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/centroids.tdb'  \
    --index_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/index.tdb'      \
    --parts_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/parts.tdb'      \
    --ids_uri           's3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/ids.tdb'        \
    --query_uri         's3://tiledb-andrew/sift/sift_query'                                       \
    --groundtruth_uri   's3://tiledb-andrew/sift/sift_groundtruth'                                 \
    --log               'ivf-shakedown.log'                                                        \
    --nqueries          0 \
    -v -d 

${ivf_hack} \
    --db_uri            's3://tiledb-andrew/sift/sift_base'                                        \
    --centroids_uri     's3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/centroids.tdb'  \
    --index_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index.tdb'      \
    --parts_uri         's3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/parts.tdb'      \
    --ids_uri           's3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/ids.tdb'        \
    --query_uri         's3://tiledb-andrew/sift/sift_query'                                       \
    --groundtruth_uri   's3://tiledb-andrew/sift/sift_groundtruth'                                 \
    --log               'ivf-shakedown.log'                                                        \
    --nqueries          0 \
    -v -d 
