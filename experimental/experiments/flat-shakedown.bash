#!/bin/bash -f

flat="${HOME}/feature-vector-prototype/experimental/cmake-build-release/src/flat"


# ${flat} \
#     --db_uri        's3://tiledb-andrew/sift/sift_base'                                        \
#     --q_uri         's3://tiledb-andrew/sift/sift_query'                                       \
#     --g_uri         's3://tiledb-andrew/sift/sift_groundtruth'                                 \
#     -v -d --log flat-shakedown.log

${flat} \
    --db_uri        's3://tiledb-andrew/sift/bigann10M_base'		                       \
    --q_uri         's3://tiledb-andrew/kmeans/benchmark/query_public_10k' \
    --g_uri         's3://tiledb-andrew/kmeans/benchmark/bigann_10M_GT_nnids' \
    --block         1000000                                                                    \
    -V -v -d --log flat-shakedown.log --order vq_heap --nqueries 100

${flat} \
    --db_uri        's3://tiledb-andrew/sift/bigann100M_base'		                       \
    --q_uri         's3://tiledb-andrew/kmeans/benchmark/query_public_10k' \
    --g_uri         's3://tiledb-andrew/kmeans/benchmark/bigann_100M_GT_nnids' \
    --block         5000000                                                                    \
    -V -v -d --log flat-shakedown.log --order vq_heap --nqueries 100


exit

${flat} \
    --db_uri        's3://tiledb-andrew/sift/bigann100M_base'		                       \
    --q_uri         's3://tiledb-andrew/sift/sift_query'                                       \
    --block         1000000                                                                   \
    -v -d --log flat-shakedown.log

${flat} \
    --db_uri        's3://tiledb-andrew/sift/bigann1B_base'		                       \
    --q_uri         's3://tiledb-andrew/sift/sift_query'                                       \
    --block         1000000                                                                   \
    -v -d --log flat-shakedown.log
