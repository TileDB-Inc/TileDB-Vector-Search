#!/bin/bash

. ./env.bash

# echo ${local_sift_prefix}

function clean_s3_arrays () {

for file in ids index parts ;
do
    echo ${local_kmeans_prefix}/${file}
    echo ${ivf_hack_prefix}/${file}
    echo ${arch_prefix}/${file}

    aws s3 ls ${ivf_hack_prefix}/${file}/
    aws s3 ls ${arch_prefix}/${file}/

    aws s3 rm ${ivf_hack_prefix}/${file}/ --recursive
    aws s3 rm ${arch_prefix}/${file}/ --recursive

done
}

init_paths us-east-1 s3://tiledb-andrew m1
clean_s3_arrays

init_paths us-east-1 s3://tiledb-andrew x86
clean_s3_arrays

init_paths us-west-2 s3://tiledb-lums m1
clean_s3_arrays

init_paths us-west-2 s3://tiledb-lums x86
clean_s3_arrays
