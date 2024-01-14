#!/bin/bash

# WIP to supersede setup.bash to reflect new file organization

################################################################
#
# Specify arrays and array locations
#
################################################################

benchmark_root=`dirname "${BASH_SOURCE[0]}"`
test_data_root="${benchmark_root}/../../external/test_data"
test_array_root="${test_data_root}/arrays"

small_arrays=("siftsmall" "siftsmall_uint8" "bigann10k" "fmnistsmall")

for small_array in "${small_arrays[@]}"; do
    small_array_path="${test_array_root}/${small_array}"
    ls ${small_array_path}
    echo "small_array_path: ${small_array_path}"

    small_array_root="${test_array_root}/${small_array}/"
    small_group_root="${small_array_root}/group"

    small_group_uri="${small_array}_group_uri"
    small_inputs_uri="${small_array}_inputs_uri"
    small_centroids_uri="${small_array}_centroids_uri"
    small_index_uri="${small_array}_index_uri"
    small_ids_uri="${small_array}_ids_uri"
    small_parts_uri="${small_array}_parts_uri"
    small_query_uri="${small_array}_query_uri"
    small_groundtruth_uri="${small_array}_groundtruth_uri"
    small_group_centroids_uri="${small_array}_group_centroids_uri"
    small_group_index_uri="${small_array}_group_index_uri"
    small_group_ids_uri="${small_array}_group_ids_uri"
    small_group_parts_uri="${small_array}_group_parts_uri"

    eval "${small_group_uri}=\"${small_array_root}/group\""
    eval "${small_inputs_uri}=\"${small_array_root}/input_vectors\""
    eval "${small_centroids_uri}=\"${small_array_root}/partition_centroids\""
    eval "${small_index_uri}=\"${small_array_root}/partition_indexes\""
    eval "${small_ids_uri}=\"${small_array_root}/shuffled_vector_ids\""
    eval "${small_parts_uri}=\"${small_array_root}/shuffled_vectors\""
    eval "${small_query_uri}=\"${small_array_root}/queries\""
    eval "${small_groundtruth_uri}=\"${small_array_root}/groundtruth\""
    eval "${small_group_centroids_uri}=\"${small_group_root}/partition_centroids\""
    eval "${small_group_index_uri}=\"${small_group_root}/partition_indexes\""
    eval "${small_group_ids_uri}=\"${small_group_root}/shuffled_vector_ids\""
    eval "${small_group_parts_uri}=\"${small_group_root}/shuffled_vectors\""

    echo small_group_uri: ${small_group_uri}
    eval echo small_group_root: \${${small_group_root}}

    small_array_arrays="${small_array}_arrays"
    eval "${small_array_arrays}=(\"${small_group_uri}\" \"${small_inputs_uri}\" \"${small_centroids_uri}\" \"${small_index_uri}\" \"${small_ids_uri}\" \"${small_parts_uri}\" \"${small_query_uri}\" \"${small_groundtruth_uri}\" \"${small_group_centroids_uri}\" \"${small_group_index_uri}\" \"${small_group_ids_uri}\" \"${small_group_parts_uri}\")"
    eval "echo \"${small_array_arrays}: \${${small_array_arrays}[@]}\""

done


################################################################
#
# Define some utility functions
#
################################################################

function print_one_schema() {
    printf "================================================================\n"
    printf "=\n= ${1}\n=\n"
    python3 -c """
import tiledb
a = tiledb.open(\"${1}\")
print(a.schema)
"""
}

function print_all_schemas() {
    for array in "${small_arrays[@]}"; do  #siftsmall, siftsmall_uint8, bigann10k, fmnistsmall
        echo "array: ${array}"
        array_arrays="${array}_arrays"
        echo "array_arrays: ${array_arrays}"
        eval this_array_arrays=(\"\${${array_arrays}[@]}\")
        echo "this_array_arrays: ${this_array_arrays[@]}"

       for uri in "${this_array_arrays[@]}"; do
            echo "uri: ${uri}"
            eval this_uri=\${${uri}}
            if [ -d "${this_uri}" ]; then
                print_one_schema "${this_uri}"
            else
                echo "uri: ${this_uri} does not exist"
            fi

        done

    done
}



################################################################
#
# Set specific array set as benchmarking set
#
################################################################


################################################################
#
# Specify benchmarking executables
#
################################################################


################################################################
#
# Specify benchmarking commands
#
################################################################



