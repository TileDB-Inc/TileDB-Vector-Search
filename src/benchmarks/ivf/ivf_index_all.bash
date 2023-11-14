#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

. ${SCRIPT_DIR}/ivf_init.bash

// siftsmall_base siftsmall_learn sift_base sift_learn 1M 10M



for id_type in 32 64; do
for px_type in 32 64; do

pref="${IVFPATH}/ivf_index  --idtype uint${id_type} --pxtype uint${px_type} \
    --num_clusters 0 -v -d --log - --force"

db_uri=${GP3}/siftsmall/siftsmall_base
index_name=${GP3}/siftsmall/flatIVF_index_siftsmall_base_${id_type}_${px_type}
cmd="${pref} --db_uri ${db_uri} --index_uri ${index_name} --ftype float"
echo ${cmd}
time ${cmd}


done
done

for id_type in 32 64; do
for px_type in 32 64; do

pref="${IVFPATH}/ivf_index  --idtype uint${id_type} --pxtype uint${px_type} \
    --num_clusters 0 -v -d --log - --force"


db_uri=${GP3}/sift/sift_base
index_name=${GP3}/sift/flatIVF_index_sift_base_${id_type}_${px_type}
cmd="${pref} --db_uri ${db_uri} --index_uri ${index_name} --ftype float"
echo ${cmd}
time ${cmd}

done
done

for id_type in 32 64; do
for px_type in 32 64; do

pref="${IVFPATH}/ivf_index  --idtype uint${id_type} --pxtype uint${px_type} \
    --num_clusters 0 -v -d --log - --force"


db_uri=${GP3}/1M/bigann1M_base
index_name=${GP3}/1M/flatIVF_index_1M_base_${id_type}_${px_type}
cmd="${pref} --db_uri ${db_uri} --index_uri ${index_name} --ftype uint8"
echo ${cmd}
time ${cmd}
done
done

for id_type in 32 64; do
for px_type in 32 64; do

pref="${IVFPATH}/ivf_index  --idtype uint${id_type} --pxtype uint${px_type} \
    --num_clusters 0 -v -d --log - --force"

db_uri=${GP3}/10M/bigann10M_base
index_name=${GP3}/10M/flatIVF_index_10M_base_${id_type}_${px_type}
cmd="${pref} --db_uri ${db_uri} --index_uri ${index_name} --ftype uint8"
echo ${cmd}
time ${cmd}


done
done
