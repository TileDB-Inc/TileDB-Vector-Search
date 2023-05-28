#!/bin/bash -f

flat="${HOME}/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/flat"

sift=s3://tiledb-andrew/sift
sift=/Users/lums/TileDB/feature-vector-prototype/external/data/arrays/sift/sift

function run() {
    order=$1
    block=$3
    nqueries=$4

    if [[ $2 -eq 1 ]]; then
	ith="--nth"
    fi
echo      '${sift}/sift_base'                                        
${flat} \
    --db_uri        "${sift}_base"                                        \
    --q_uri         "${sift}_query"                                       \
    --g_uri         "${sift}_groundtruth"                                 \
    --order         ${order}                                                   \
    ${ith}                                                                     \
    --block         ${block}                                                   \
    --nthreads      8                                                          \
    -v -d --log method-shakedown.log
}


for nqueries in 1 100 0;
do
    for order in vq_heap qv_heap;
    do
	run ${order} " " 0 ${nqueries}
	run ${order} " " 0 ${nqueries}
    done
    
    for order in vq_nth qv_nth gemm;
    do
	for nth in 1 0;
	do
	    run ${order} ${nth} 0 ${nqueries}
	    run ${order} ${nth} 0 ${nqueries}
	done
    done
    
    for block in 1000 10000 100000;
    do
	for nth in 1 0;
	do
	    run gemm ${nth} ${block} ${nqueries}
	done
    done
done


