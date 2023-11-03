#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

. ${SCRIPT_DIR}/ivf_init.bash

if [ ! -e ${SIFT_QUERY} ]; then
  echo "File ${SIFT_QUERY} does not exist!"
fi
if [ ! -e ${SIFT_INDEX} ]; then
  echo "File ${SIFT_INDEX} does not exist!"
fi
if [ ! -e ${SIFT_GROUNDTRUTH} ]; then
  echo "File ${SIFT_GROUNDTRUTH} does not exist!"
fi

cmd="${IVFPATH}/ivf_query  --index_uri ${SIFT_INDEX} --query_uri ${SIFT_QUERY}  --groundtruth_uri ${SIFT_GROUNDTRUTH}  -k 10 -v -d --log -"
echo ${cmd}
time ${cmd}