#!/bin/bash -f

flat="${HOME}/feature-vector-prototype/experimental/cmake-build-release/src/flat"


${flat} \
    --db_uri        's3://tiledb-andrew/sift/sift_base'                                        \
    --q_uri         's3://tiledb-andrew/sift/sift_query'                                       \
    --g_uri         's3://tiledb-andrew/sift/sift_groundtruth'                                 \
    -v -d --log flat-shakedown.log

${flat} \
    --db_uri        's3://tiledb-andrew/sift/bigann10M_base'		                       \
    --q_uri         's3://tiledb-andrew/sift/sift_query'                                       \
    --block         1000000                                                                    \
    -v -d --log flat-shakedown.log

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
