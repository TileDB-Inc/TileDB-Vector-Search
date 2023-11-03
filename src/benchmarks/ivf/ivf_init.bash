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

export SRCROOT=${VECTOR_SEARCH}/src/

export SIFT=siftsmall
export SIFTPATH=${VECTOR_SEARCH}/external/data/gp3/${SIFT}
export DATAPATH=${SIFTPATH}

export IVFPATH=${SRCROOT}/cmake-build-relwithdebinfo/libtiledbvectorsearch/src/ivf/

export SIFT_LEARN=${SIFTPATH}/${SIFT}_base
export SIFT_QUERY=${SIFTPATH}/${SIFT}_query
export SIFT_GROUNDTRUTH=${SIFTPATH}/${SIFT}_groundtruth

export SIFT_INDEX=${SIFTPATH}/flatIVF_index_${SIFT}_base

export ID_TYPE=uint32
export PX_TYPE=uint64
export F_TYPE=float
