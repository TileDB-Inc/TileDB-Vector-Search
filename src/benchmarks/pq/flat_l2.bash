#!/bin/bash

export VECTOR_SEARCH=${HOME}/TileDB/TileDB-Vector-Search

if [ ! -d ${VECTOR_SEARCH} ]; then
  echo "${VECTOR_SEARCH} does not exist"
  export VECTOR_SEARCH=${HOME}/TileDB-Vector-Search
else
  if [ ! -d ${VECTOR_SEARCH} ]; then
    echo "${VECTOR_SEARCH} does not exist"
    return 1
  fi
fi
echo "VECTOR_SEARCH is ${VECTOR_SEARCH}"

export PQROOT=${VECTOR_SEARCH}/src/

export SIFT=sift
export SIFTPATH=${VECTOR_SEARCH}/external/data/gp3/${SIFT}
export DATAPATH=${SIFTPATH}

export PQPATH=${PQROOT}/cmake-build-relwithdebinfo/libtiledbvectorsearch/src/

export SIFT_LEARN=${SIFTPATH}/${SIFT}_base
export SIFT_QUERY=${SIFTPATH}/${SIFT}_query
export SIFT_GROUNDTRUTH=${SIFTPATH}/${SIFT}_groundtruth

if [ ! -e ${SIFT_LEARN} ]; then
  echo "File ${SIFT_LEARN} does not exist!"
fi
if [ ! -e ${SIFT_QUERY} ]; then
  echo "File ${SIFT_QUERY} does not exist!"
fi
if [ ! -e ${SIFT_GROUNDTRUTH} ]; then
  echo "File ${SIFT_GROUNDTRUTH} does not exist!"
fi

cmd="${PQPATH}/flat_l2 --db_uri ${SIFT_LEARN} --query_uri ${SIFT_QUERY} --groundtruth_uri ${SIFT_GROUNDTRUTH} --alg qv -v -d --log -"
echo ${cmd}
time ${cmd}
