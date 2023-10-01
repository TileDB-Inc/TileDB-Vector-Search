#!/bin/bash

export VECTOR_SEARCH=${HOME}/TileDB/TileDB-Vector-Search

export DISKANNROOT=${VECTOR_SEARCH}/external/DiskANN

export SIFT=siftsmall
export SIFTPATH=${VECTOR_SEARCH}/external/data/bins/${SIFT}
export DATAPATH=${SIFTPATH}

export DISKANNPATH=${DISKANNROOT}/build/apps

export SIFT_LEARN=${SIFTPATH}/${SIFT}_learn.fbin
export SIFT_QUERY=${SIFTPATH}/${SIFT}_query.fbin
export SIFT_INDEX=${SIFTPATH}/index_${SIFT}_learn_R${R}_L${L}_A1.2
export SIFT_GROUNDTRUTH=${SIFTPATH}/${SIFT}_groundtruth.ibin


Rs=(32 64)
Ls=(50 100)



for R in ${Rs[@]};
do
    for L in ${Ls[@]};
    do

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


for R in ${Rs[@]};
do
    for L in ${Ls[@]};
    do

echo	${DISKANNPATH}/build_memory_index  --data_type float --dist_fn l2 --data_path ${SIFT_LEARN}  --index_path_prefix ${SIFT_INDEX} -R ${R} -L ${L} --alpha 1.2
    done
done

for R in ${Rs[@]};
do
    for L in ${Ls[@]};
    do
echo	${DISKANNPATH}/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix ${SIFT_INDEX} --query_file ${SIFT_QUERY}  --gt_file ${SIFT_GROUNDTRUTH}  -K 10 -L 10 20 30 40 50 100 --result_path ${SIFTPATH}/res
    done
done
