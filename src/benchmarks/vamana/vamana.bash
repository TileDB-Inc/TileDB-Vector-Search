#!/bin/bash

benchmark_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VECTOR_SEARCH="$(dirname "$(dirname "$(dirname "$benchmark_path")")")"

if [ ! -d ${VECTOR_SEARCH} ]; then
    echo "${VECTOR_SEARCH} not found"
    export VECTOR_SEARCH=${HOME}/TileDB/TileDB-Vector-Search-complete-index-vamana
    echo "Trying ${VECTOR_SEARCH}"
fi
if [ ! -d ${VECTOR_SEARCH} ]; then
	echo "${VECTOR_SEARCH} does not exist"
	return 1
fi
echo "VECTOR_SEARCH is ${VECTOR_SEARCH}"

export VAMANAROOT=${VECTOR_SEARCH}/src/

export SIFT=siftsmall
export SIFTPATH=${VECTOR_SEARCH}/external/test_data/arrays/${SIFT}
export DATAPATH=${SIFTPATH}

export VAMANAPATH=${VAMANAROOT}/cmake-build-relwithdebinfo/libtiledbvectorsearch/src/vamana

export SIFT_LEARN=${SIFTPATH}/input_vectors
export SIFT_QUERY=${SIFTPATH}/queries
export SIFT_GROUNDTRUTH=${SIFTPATH}/groundtruth

# echo ${VAMANAPATH}/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file ${SIFT_LEARN} --query_file  ${SIFT_QUERY} --gt_file ${SIFT_GROUNDTRUTH} --K 100

# Rs=(32 64)
# Ls=(50 100)
# Ts=(1 `nproc`)

Rs=(64)
Ls=(100)
Ts=(1)

for R in ${Rs[@]}; do
    for L in ${Ls[@]}; do
	export SIFT_INDEX=${SIFTPATH}/index_${SIFT}_learn_R${R}_L${L}_A1.2
	if [ ! -e ${SIFT_LEARN} ]; then
	    echo "File ${SIFT_LEARN} does not exist!"
	fi
	if [ ! -e ${SIFT_QUERY} ]; then
	    echo "File ${SIFT_QUERY} does not exist!"
	fi
	if [ ! -e ${SIFT_INDEX} ]; then
	    echo "File ${SIFT_INDEX} does not exist!"
	fi
	if [ ! -e ${SIFT_GROUNDTRUTH} ]; then
	    echo "File ${SIFT_GROUNDTRUTH} does not exist!"
	fi
    done
done


for T in ${Ts[@]}; do
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "================================================================"
    for R in ${Rs[@]}; do
	echo "----------------------------------------------------------------"
	for L in ${Ls[@]}; do
	    export SIFT_INDEX=${SIFTPATH}/index_${SIFT}_learn_R${R}_L${L}_A1.2
	    cmd="${VAMANAPATH}/vamana_index --inputs_uri ${SIFT_LEARN}  --index_uri ${SIFT_INDEX} -R ${R} -L ${L} --alpha 1.2 --nthreads ${T} -v -d --log - --force"
	    echo ${cmd}
	    time ${cmd}
	done
    done
    echo "================================================================"
    for R in ${Rs[@]}; do
	echo "----------------------------------------------------------------"
	for L in ${Ls[@]}; do
	    export SIFT_INDEX=${SIFTPATH}/index_${SIFT}_learn_R${R}_L${L}_A1.2
	    cmd="${VAMANAPATH}/vamana_query  --index_uri ${SIFT_INDEX} --query_uri ${SIFT_QUERY}  --groundtruth_uri ${SIFT_GROUNDTRUTH}  -k 10 -L ${L} --nthreads ${T} -v -d --log -"
	    echo ${cmd}
	    time ${cmd}
	done
    done
done
