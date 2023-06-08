#!/bin/bash

declare -g gp3_root=/home/lums/feature-vector-prototype/experimental/external/data/gp3
declare -g nvme_root=/mnt/ssd

function init_1M () {
    declare -g db_uri="s3://tiledb-andrew/sift/bigann1M_base"
    declare -g centroids_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/centroids.tdb"
    declare -g parts_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/parts.tdb"
    declare -g index_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/index.tdb"
    declare -g ids_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/ids.tdb"
    declare -g query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    declare -g groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_1M_GT_nnids"
}

function init_1M_gp3 () {
    declare -g db_uri="${gp3_root}/1M/bigann1M_base"
    declare -g centroids_uri="${gp3_root}/1M/centroids.tdb"
    declare -g parts_uri="${gp3_root}/1M/parts.tdb"
    declare -g index_uri="${gp3_root}/1M/index.tdb"
    declare -g ids_uri="${gp3_root}/1M/ids.tdb"
    declare -g query_uri="${gp3_root}/1M/query_public_10k"
    declare -g groundtruth_uri="${gp3_root}/1M/bigann_1M_GT_nnids"
}

function init_1M_nvme () {
    declare -g db_uri="${nvme_root}/1M/bigann1M_base"
    declare -g centroids_uri="${nvme_root}/1M/centroids.tdb"
    declare -g parts_uri="${nvme_root}/1M/parts.tdb"
    declare -g index_uri="${nvme_root}/1M/index.tdb"
    declare -g ids_uri="${nvme_root}/1M/ids.tdb"
    declare -g query_uri="${nvme_root}/1M/query_public_10k"
    declare -g groundtruth_uri="${nvme_root}/1M/bigann_1M_GT_nnids"
}

function init_10M () {
    declare -g db_uri="s3://tiledb-andrew/sift/bigann10M_base"
    declare -g centroids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/centroids.tdb"
    declare -g parts_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/parts.tdb"
    declare -g index_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/index.tdb"
    declare -g ids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/ids.tdb"
    declare -g query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    declare -g groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_10M_GT_nnids"
}

function init_10M_gp3 () {
    declare -g db_uri="${gp3_root}/10M/bigann10M_base"
    declare -g centroids_uri="${gp3_root}/10M/centroids.tdb"
    declare -g parts_uri="${gp3_root}/10M/parts.tdb"
    declare -g index_uri="${gp3_root}/10M/index.tdb"
    declare -g ids_uri="${gp3_root}/10M/ids.tdb"
    declare -g query_uri="${gp3_root}/10M/query_public_10k"
    declare -g groundtruth_uri="${gp3_root}/10M/bigann_10M_GT_nnids"
}

function init_10M_nvme () {
    declare -g db_uri="${nvme_root}/10M/bigann10M_base"
    declare -g centroids_uri="${nvme_root}/10M/centroids.tdb"
    declare -g parts_uri="${nvme_root}/10M/parts.tdb"
    declare -g index_uri="${nvme_root}/10M/index.tdb"
    declare -g ids_uri="${nvme_root}/10M/ids.tdb"
    declare -g query_uri="${nvme_root}/10M/query_public_10k"
    declare -g groundtruth_uri="${nvme_root}/10M/bigann_10M_GT_nnids"
}

function init_100M () {
    declare -g db_uri="s3://tiledb-andrew/sift/bigann100M_base"
    declare -g centroids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/centroids.tdb"
    declare -g parts_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/parts.tdb"
    declare -g index_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/index.tdb"
    declare -g ids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/ids.tdb"
    declare -g query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    declare -g groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_100M_GT_nnids"
}

function init_100M_gp3 () {
    declare -g db_uri="${gp3_root}/100M/bigann100M_base"
    declare -g centroids_uri="${gp3_root}/100M/centroids.tdb"
    declare -g parts_uri="${gp3_root}/100M/parts.tdb"
    declare -g index_uri="${gp3_root}/100M/index.tdb"
    declare -g ids_uri="${gp3_root}/100M/ids.tdb"
    declare -g query_uri="${gp3_root}/100M/query_public_10k"
    declare -g groundtruth_uri="${gp3_root}/100M/bigann_100M_GT_nnids"
}

function init_100M_nvme () {
    declare -g db_uri="${nvme_root}/100M/bigann100M_base"
    declare -g centroids_uri="${nvme_root}/100M/centroids.tdb"
    declare -g parts_uri="${nvme_root}/100M/parts.tdb"
    declare -g index_uri="${nvme_root}/100M/index.tdb"
    declare -g ids_uri="${nvme_root}/100M/ids.tdb"
    declare -g query_uri="${nvme_root}/100M/query_public_10k"
    declare -g groundtruth_uri="${nvme_root}/100M/bigann_100M_GT_nnids"
}

function init_1B () {
    declare -g db_uri="s3://tiledb-nikos/vector-search/datasets/arrays/sift-1b-col-major"
    declare -g centroids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/centroids.tdb"
    declare -g parts_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/parts.tdb"
    declare -g index_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index.tdb"
    declare -g ids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/ids.tdb"
    declare -g query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    declare -g groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_1B_GT_nnids"
}

function init_1B_gp3 () {
    declare -g db_uri="${gp3_root}/1B/sift-1b-col-major"
    declare -g centroids_uri="${gp3_root}/1B/centroids.tdb"
    declare -g parts_uri="${gp3_root}/1B/parts.tdb"
    declare -g index_uri="${gp3_root}/1B/index.tdb"
    declare -g ids_uri="${gp3_root}/1B/ids.tdb"
    declare -g query_uri="${gp3_root}/1B/query_public_10k"
    declare -g groundtruth_uri="${gp3_root}/1B/bigann_1B_GT_nnids"
}

function init_1B_nvme () {
    declare -g db_uri="${nvme_root}/1B/sift-1b-col-major"
    declare -g centroids_uri="${nvme_root}/1B/centroids.tdb"
    declare -g parts_uri="${nvme_root}/1B/parts.tdb"
    declare -g index_uri="${nvme_root}/1B/index.tdb"
    declare -g ids_uri="${nvme_root}/1B/ids.tdb"
    declare -g query_uri="${nvme_root}/1B/query_public_10k"
    declare -g groundtruth_uri="${nvme_root}/1B/bigann_1B_GT_nnids"
}

function verify_s3 () {
    aws s3 ls ${db_uri}
    aws s3 ls ${centroids_uri}
    aws s3 ls ${parts_uri}
    aws s3 ls ${index_uri}
    aws s3 ls ${ids_uri}
    aws s3 ls ${query_uri}
    aws s3 ls ${groundtruth_uri}
}

function verify_all () {
    init_1M
    verify_s3
    init_10M
    verify_s3
    init_100M
    verify_s3
    init_1B
    verify_s3
}

function print_one_schema() {
    printf "================================================================\n"
    printf "=\n= ${1}\n=\n"
    python3 -c """
import tiledb
a = tiledb.open(\"${1}\")
print(a.schema)
"""
}

function print_seven_schemas () {
    print_one_schema ${db_uri}
    print_one_schema ${centroids_uri}
    print_one_schema ${parts_uri}
    print_one_schema ${index_uri}
    print_one_schema ${ids_uri}
    print_one_schema ${query_uri}
    print_one_schema ${groundtruth_uri}
}

function print_all_schemas () {
    init_1M
    print_seven_schemas
    init_10M
    print_seven_schemas
    init_100M
    print_seven_schemas

    if [ -d "${gp3_root}" ]; then
	init_1M_gp3
	print_seven_schemas
	init_10M_gp3
	print_seven_schemas
	init_100M_gp3
	print_seven_schemas
	init_1B_gp3
	print_seven_schemas
    else
	printf "\n++\n++ ${gp3_root} not found\n++\n"
    fi

    if [ -d "${nvme_root}" ]; then
	init_1M_nvme
	print_seven_schemas
	init_10M_nvme
	print_seven_schemas
	init_100M_nvme
	print_seven_schemas
	init_1B
	print_seven_schemas
	init_1B_gp3
	print_seven_schemas
    else
	printf "\n++\n++ ${nvme_root} not found\n++\n"
    fi

}

function ivf_query() {

    while [ "$#" -gt 0 ]; do
	case "$1" in
	    -x|--exec)
		ivf_query=${2}
		shift 2
		;;
	    -h|--help)
		shift 1
		;;
	    -d|--debug)
		debug="-d"
		shift 1
		;;
	    -v|--verbose)
		verbose="-v"
		shift 1
		;;
	    --k|--knn|--k_nn)
		k_nn="--k ${2}"
		shift 2
		;;
	    --nqueries)
		nqueries="--nqueries ${2}"
		shift 2
		;;
	    --nthreads)
		nthreads="--nthreads ${2}"
		shift 2
		;;
	    --cluster|--nprobe)
		cluster="--cluster ${2}"
		shift 2
		;;
	    --block|--blocksize)
		blocksize="--blocksize ${2}"
		shift 2
		;;
	    --finite)
		finite="--finite"
		shift 1
		;;
	    --log)
		log="--log ${2}"
		shift 2
		;;
	    *)
		echo "Unknown option: $1"
		return 1
		;;
	esac
    done

    if [ -z "${ivf_query}" ];
    then
	echo "ivf_query executable not set"
	return -1
    fi

    query="\
${ivf_query} \
--db_uri ${db_uri} \
--centroids_uri ${centroids_uri} \
--parts_uri ${parts_uri} \
--index_uri ${index_uri} \
--ids_uri ${ids_uri} \
--query_uri ${query_uri} \
--groundtruth_uri ${groundtruth_uri} \
${k_nn} \
${nqueries} \
${nthreads} \
${cluster} \
${blocksize} \
${finite} \
${log} \
${verbose} \
${debug}"

    printf "================================================================\n"
    printf "=\n=\n"
    printf "${query}\n"
    eval "${query}"
}
