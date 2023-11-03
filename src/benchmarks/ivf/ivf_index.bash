#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

. ${SCRIPT_DIR}/ivf_init.bash

if [ ! -e ${SIFT_LEARN} ]; then
  echo "File ${SIFT_LEARN} does not exist!"
fi


cmd="${IVFPATH}/ivf_index --db_uri ${SIFT_LEARN}  --ftype ${F_TYPE} --index_uri ${SIFT_INDEX} --idtype ${ID_TYPE} --pxtype ${PX_TYPE} -v -d --log - --force"
echo ${cmd}
time ${cmd}

# cmd="${IVFPATH}/ivf_query  --index_uri ${SIFT_INDEX} --query_uri ${SIFT_QUERY}  --groundtruth_uri ${SIFT_GROUNDTRUTH}  -k 10 -v -d --log -"
# echo ${cmd}
# time ${cmd}
