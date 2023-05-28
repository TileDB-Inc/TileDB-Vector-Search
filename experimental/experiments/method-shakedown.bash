#!/bin/bash -f

flat="${HOME}/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/flat"

sift=s3://tiledb-andrew/sift
sift=/Users/lums/TileDB/feature-vector-prototype/external/data/arrays/sift

function run() {
    order=$1
    nth=$2
    block=$3
    nqueries=$4
echo      '${sift}/sift_base'                                        
${flat} \
    --db_uri        "${sift}/sift_base"                                        \
    --q_uri         "${sift}/sift_query"                                       \
    --g_uri         "${sift}/sift_groundtruth"                                 \
    --order         ${order}                                                   \
    ${nth}                                                                     \
    --block         ${block}                                                   \
    --nthreads      8                                                          \
    -v -d --log method-shakedown.log
}


for nqueries in 1 100 0;
do
for order in vq_heap qv_heap;
do
    run ${order} "" 0
done

for order in vq_nth qv_nth gemm;
do
    for nth in "-nth" "";
    do
	run ${order} ${nth} 0
    done
done

for block in 1000 10000 100000;
do
    for nth in "--nth" ""
    do
	run gemm ${nth} ${block}
    done
done
done


