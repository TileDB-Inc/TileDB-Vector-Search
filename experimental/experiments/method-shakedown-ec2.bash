#!/bin/bash -f

flat="${HOME}/feature-vector-prototype/experimental/cmake-build-release/src/flat"

sift=s3://tiledb-andrew/sift
sift=${HOME}/feature-vector-prototype/external/data/arrays/sift/sift

function run() {
    order=$1
    block=$3
    nqueries=$4
    nthreads=$5

    if [[ $2 -eq 1 ]]; then
	ith="--nth"
    else
	ith=""
    fi

echo ${flat} \
    --db_uri        "${sift}_base"                                        \
    --q_uri         "${sift}_query"                                       \
    --g_uri         "${sift}_groundtruth"                                 \
    --order         ${order}                                                   \
    ${ith}                                                                     \
    --block         ${block}                                                   \
    --nthreads      ${nthreads}                                                          \
    --nqueries      ${nqueries}                                                          \
    -V -v -d --log method-shakedown-ec2.json

 ${flat} \
    --db_uri        "${sift}_base"                                        \
    --q_uri         "${sift}_query"                                       \
    --g_uri         "${sift}_groundtruth"                                 \
    --order         ${order}                                                   \
    ${ith}                                                                     \
    --block         ${block}                                                   \
    --nthreads      ${nthreads}                                                          \
    --nqueries      ${nqueries}                                                          \
    -V -v -d --log method-shakedown-ec2.json
}


for nqueries in 1 100 0;
do
    for nthreads in 96 64 48 32 16 8 4 2 1;
    do
	for order in vq_heap ;#qv_heap;
	do
	    run ${order} " " 0 ${nqueries} ${nthreads}
	done
	
	run gemm 0 100000 ${nqueries} ${nthreads}
	
	# for order in vq_nth qv_nth gemm;
	# do
	# 	for nth in 1 0;
	# 	do
	# 	    run ${order} ${nth} 0 ${nqueries}
	# 	done
	# done
	
	# for block in 1000 10000 100000;
	# do
	# 	for nth in 1 0;
	# 	do
	# 	    run gemm ${nth} ${block} ${nqueries}
	# 	done
	# done
	# echo
	# echo "# [ Total ======================================="
	# echo
    done
done


