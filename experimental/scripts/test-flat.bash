#!/bin/bash -f

flat="${HOME}/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/flat"

sift=/Users/lums/TileDB/feature-vector-prototype/external/data/arrays/sift/
kmeans=/Users/lums/TileDB/feature-vector-prototype/external/data/arrays/kmeans/ivf_hack/m1/

for size in siftsmall sift;
do

    db_uri=${sift}/${size}_base
    q_uri=${sift}/${size}_query
    g_uri=${sift}/${size}_groundtruth
    
    for order in qv vq gemm;
    do
	for hardway in " " "--hardway";
	do
	    ${flat} --db_uri ${db_uri} --q_uri ${q_uri} --g_uri ${g_uri}  --order ${order} ${hardway}
	done
    done
done

