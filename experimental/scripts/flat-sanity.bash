#!/bin/bash -f

flat="${HOME}/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/flat"

sift=/Users/lums/TileDB/feature-vector-prototype/external/data/arrays/sift/
kmeans=/Users/lums/TileDB/feature-vector-prototype/external/data/arrays/kmeans/ivf_hack/m1/

# sift=s3://tiledb-andrew/sift
# kmeans=s3://tiledb-andrew/kmeans/ivf_hack/m1

# ${flat} --db_uri s3://tiledb-andrew/sift/siftsmall_base --q_uri s3://tiledb-andrew/sift/siftsmall_query --g_uri s3://tiledb-andrew/sift/siftsmall_groundtruth --order gemm



for size in sift ;#siftsmall;
do
    db_uri=${sift}/${size}_base
    q_uri=${sift}/${size}_query
    g_uri=${sift}/${size}_groundtruth
    
    for order in gemm ; do # qv vq gemm;    do
	for hardway in " " ;#"--hardway";
	do
	    ${flat} --db_uri ${db_uri} --q_uri ${q_uri} --g_uri ${g_uri} --order ${order} ${hardway} --blocked --ndb 100
#	    ${flat} --db_uri ${db_uri} --q_uri ${q_uri} --g_uri ${g_uri} --order ${order} ${hardway} --blocked --ndb 10000
#	    ${flat} --db_uri ${db_uri} --q_uri ${q_uri} --g_uri ${g_uri} --order ${order} ${hardway} --blocked --ndb 50000
	done
    done
done

