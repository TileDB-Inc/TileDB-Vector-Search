#!/bin/bash

###############################################################################
#
# This is a convenience script that sets up functions to invoke the C++
# CLI programs with the pre-defined TileDB arrays for the bigann benchmarks
#
# After sourcing this script you can configure the set of required arrays
# by invoking (e.g.) `init_1B_gp3` which will initialize variables for
# all of the arrays necssary to benchmark with the bigann 1B dataset, stored
# on your local filesystem.  Similar commands exist to initialize for the
# 1M, 10M, and 100M subsets of the 1B dataset.  In addition, you can
# substitute `s3` for `gp3` if you want to use the arrays stored in S3.
# Note that you will need to download the arrays to local storage and
# point to their location with the `ec2_root` or `m1_root` variable.
#
# Once you have initialized the dataset you are going to use, you can
# invoke the benchmark you want with `ivf_query` or `flat_query`
#
# The following will run the `flat_l2` benchmark using the 10M dataset
# stored in S3:
# ```
# $ . ./setup.bash
# $ init_10M_s3
# $ flat_query
# ```
#
# The `ivf_query` and `flat_query` functions will also accept many of the
# available options for the driver programs themselves.
#
###############################################################################
#
# You will need to edit the variables in this section

# AWS information
# Defines the variables instance_id, volume_id, region, and instance_ip
# 
if [ -f ~/.bash_awsrc ]; then
    . ~/.bash_awsrc
fi
#
# You can define them manually here rather than in a file by defining these:
# instance_id="i-01234567890abcdef"
# volume_id="vol-abcdefghihklmnopq"
# region="us-east-1"
# instance_ip=$(aws ec2 describe-instances --instance-ids i-0daab006867136323 --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
#

# Define check_instance_status if it is not already defined
if [[ "$(type -t check_instance_status)" != "function" ]]; then
    check_instance_status() {
	local local_instance_ip=${1:-${instance_ip}}
	local ssh_port=22
	local local_max_tries=${max_nc_tries:-5}
	local sleep_time=${nc_tries_sleep:-5}
	local timeout=${nc_timeout:-2}

	local tries=0
	while [ $tries -lt $local_max_tries ]; do

	    if ! host "${local_instance_ip}" >/dev/null 2>&1; then
		echo "Invalid host name: $local_instance_ip"
		return 1  # Exit the function with a non-zero status
	    fi

	    nc -z -G ${timeout} $local_instance_ip $ssh_port >/dev/null 2>&1
	    if [ $? -ne 0 ]; then
		tries=$((tries + 1))
		if [ $tries -lt $local_max_tries ];
		then
		    echo "Instance is not ready for SSH connection yet."
		    sleep $sleep_time
		fi
	    else
		echo "Instance is ready for SSH connection."
		return 0
	    fi
	done

	echo "Instance is not ready after ${local_max_tries} tries."
	return 1
    }
fi

# Locations of the "ivf_flat" and "flat_l2" driver executables
ec2_ivf_flat="/home/lums/TileDB-Vector-Search/src/cmake-build-release/libtiledbvectorsearch/src/ivf_flat"
m1_ivf_flat="/Users/lums/TileDB/TileDB-Vector-Search/src/cmake-build-release/src/ivf_flat"
ec2_flat="/home/lums/TileDB-Vector-Search/src/cmake-build-release/src/flat_l2"
m1_flat="/Users/lums/TileDB/TileDB-Vector-Search/src/cmake-build-release/src/flat_l2"

###############################################################################

if [ -f "${ivf_query}" ]; then
    ivf_query="${ivf_query}"
elif [ -f "${ec2_ivf_flat}" ]; then
    ivf_query="${ec2_ivf_flat}"
elif [ -f "${m1_ivf_flat}" ]; then
    ivf_query="${m1_ivf_flat}"
else
    echo "Neither ivf_flat executable file exists"
fi

if [ -f "${flat_query}" ]; then
    flat_query="${flat_query}"
elif [ -f "${ec2_flat}" ]; then
    flat_query="${ec2_flat}"
elif [ -f "${m1_flat}" ]; then
    flat_query="${m1_flat}"
else
    echo "Neither flat executable file exists"
fi

nvme_root=/mnt/ssd

ec2_root="/home/lums/TileDB-Vector-Search/external/data/gp3"
m1_root="/Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3"

if [ -d "${gp3_root}" ]; then
    gp3_root=${gp3_root}
elif [ -d "${ec2_root}" ]; then
    gp3_root=${ec2_root}
elif [ -d "${m1_root}" ]; then
    gp3_root=${m1_root}
else
    echo "gp3 directory does not exist"
fi

db_uri="not_set"
centroids_uri="not_set"
parts_uri="not_set"
index_uri="not_set"
sizes_uri="not_set"
ids_uri="not_set"
query_uri="not_set"
groundtruth_uri="not_set"

function init_1M_s3 () {
    db_uri="s3://tiledb-andrew/sift/bigann1M_base"
    centroids_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/centroids.tdb"
    parts_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/parts.tdb"
    index_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/index.tdb"
    sizes_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/index_size.tdb"
    ids_uri="s3://tiledb-andrew/sift/bigann1M_base_tdb/ids.tdb"
    query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_1M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_1M_gp3 () {
    db_uri="${gp3_root}/1M/bigann1M_base"
    centroids_uri="${gp3_root}/1M/centroids.tdb"
    parts_uri="${gp3_root}/1M/parts.tdb"
    index_uri="${gp3_root}/1M/index.tdb"
    sizes_uri="${gp3_root}/1M/index_size.tdb"
    ids_uri="${gp3_root}/1M/ids.tdb"
    query_uri="${gp3_root}/1M/query_public_10k"
    groundtruth_uri="${gp3_root}/1M/bigann_1M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_1M_nvme () {
    db_uri="${nvme_root}/1M/bigann1M_base"
    centroids_uri="${nvme_root}/1M/centroids.tdb"
    parts_uri="${nvme_root}/1M/parts.tdb"
    index_uri="${nvme_root}/1M/index.tdb"
    sizes_uri="${nvme_root}/1M/index_size.tdb"
    ids_uri="${nvme_root}/1M/ids.tdb"
    query_uri="${nvme_root}/1M/query_public_10k"
    groundtruth_uri="${nvme_root}/1M/bigann_1M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_10M_s3 () {
    db_uri="s3://tiledb-andrew/sift/bigann10M_base"
    centroids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/centroids.tdb"
    parts_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/parts.tdb"
    index_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/index.tdb"
    sizes_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/index_size.tdb"
    ids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-10m-1000p/ids.tdb"
    query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_10M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_10M_gp3 () {
    db_uri="${gp3_root}/10M/bigann10M_base"
    centroids_uri="${gp3_root}/10M/centroids.tdb"
    parts_uri="${gp3_root}/10M/parts.tdb"
    index_uri="${gp3_root}/10M/index.tdb"
    sizes_uri="${gp3_root}/10M/index_size.tdb"
    ids_uri="${gp3_root}/10M/ids.tdb"
    query_uri="${gp3_root}/10M/query_public_10k"
    groundtruth_uri="${gp3_root}/10M/bigann_10M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_10M_nvme () {
    db_uri="${nvme_root}/10M/bigann10M_base"
    centroids_uri="${nvme_root}/10M/centroids.tdb"
    parts_uri="${nvme_root}/10M/parts.tdb"
    index_uri="${nvme_root}/10M/index.tdb"
    sizes_uri="${nvme_root}/10M/index_size.tdb"
    ids_uri="${nvme_root}/10M/ids.tdb"
    query_uri="${nvme_root}/10M/query_public_10k"
    groundtruth_uri="${nvme_root}/10M/bigann_10M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_100M_s3 () {
    db_uri="s3://tiledb-andrew/sift/bigann100M_base"
    centroids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/centroids.tdb"
    parts_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/parts.tdb"
    index_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/index.tdb"
    sizes_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/index_size.tdb"
    ids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-100m-10000p/ids.tdb"
    query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_100M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_100M_gp3 () {
    db_uri="${gp3_root}/100M/bigann100M_base"
    centroids_uri="${gp3_root}/100M/centroids.tdb"
    parts_uri="${gp3_root}/100M/parts.tdb"
    index_uri="${gp3_root}/100M/index.tdb"
    sizes_uri="${gp3_root}/100M/index_size.tdb"
    ids_uri="${gp3_root}/100M/ids.tdb"
    query_uri="${gp3_root}/100M/query_public_10k"
    groundtruth_uri="${gp3_root}/100M/bigann_100M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_100M_nvme () {
    db_uri="${nvme_root}/100M/bigann100M_base"
    centroids_uri="${nvme_root}/100M/centroids.tdb"
    parts_uri="${nvme_root}/100M/parts.tdb"
    index_uri="${nvme_root}/100M/index.tdb"
    sizes_uri="${nvme_root}/100M/index_size.tdb"
    ids_uri="${nvme_root}/100M/ids.tdb"
    query_uri="${nvme_root}/100M/query_public_10k"
    groundtruth_uri="${nvme_root}/100M/bigann_100M_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_1B_s3 () {
    db_uri="s3://tiledb-nikos/vector-search/datasets/arrays/sift-1b-col-major"
    centroids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/centroids.tdb"
    parts_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/parts.tdb"
    index_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index.tdb"
    sizes_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index_size.tdb"
    ids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/ids.tdb"
    query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_1B_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_1B_gp3 () {
    db_uri="${gp3_root}/1B/sift-1b-col-major"
    centroids_uri="${gp3_root}/1B/centroids.tdb"
    parts_uri="${gp3_root}/1B/parts.tdb"
    index_uri="${gp3_root}/1B/index.tdb"
    sizes_uri="${gp3_root}/1B/index_size.tdb"
    ids_uri="${gp3_root}/1B/ids.tdb"
    query_uri="${gp3_root}/1B/query_public_10k"
    groundtruth_uri="${gp3_root}/1B/bigann_1B_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_1B_nvme () {
    db_uri="${nvme_root}/1B/sift-1b-col-major"
    centroids_uri="${nvme_root}/1B/centroids.tdb"
    parts_uri="${nvme_root}/1B/parts.tdb"
    index_uri="${nvme_root}/1B/index.tdb"
    sizes_uri="${nvme_root}/1B/index_size.tdb"
    ids_uri="${nvme_root}/1B/ids.tdb"
    query_uri="${nvme_root}/1B/query_public_10k"
    groundtruth_uri="${nvme_root}/1B/bigann_1B_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_sift_s3 () {
    db_uri="s3://tiledb-nikos/vector-search/datasets/arrays/sift-1b-col-major"
    centroids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/centroids.tdb"
    parts_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/parts.tdb"
    index_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index.tdb"
    sizes_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index_size.tdb"
    ids_uri="s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/ids.tdb"
    query_uri="s3://tiledb-andrew/kmeans/benchmark/query_public_10k"
    groundtruth_uri="s3://tiledb-andrew/kmeans/benchmark/bigann_1B_GT_nnids"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_sift_gp3 () {
    db_uri="${gp3_root}/sift/sift_base"
    centroids_uri="${gp3_root}/sift/centroids"
    parts_uri="${gp3_root}/sift/parts"
    index_uri="${gp3_root}/sift/index"
    sizes_uri="${gp3_root}/sift/index_size"
    ids_uri="${gp3_root}/sift/ids"
    query_uri="${gp3_root}/sift/sift_query"
    groundtruth_uri="${gp3_root}/sift/sift_groundtruth"
    echo "     -|-  ${FUNCNAME[0]}"
}

function init_sift_nvme () {
    db_uri="${nvme_root}/sift/sift_base"
    centroids_uri="${nvme_root}/sift/centroids"
    parts_uri="${nvme_root}/sift/parts"
    index_uri="${nvme_root}/sift/index"
    index_uri="${nvme_root}/sift/index_size"
    ids_uri="${nvme_root}/sift/ids"
    query_uri="${nvme_root}/sift/sift_query"
    groundtruth_uri="${nvme_root}/sift/sift_groundtruth"
    echo "     -|-  ${FUNCNAME[0]}"
}


function verify_s3 () {
    aws s3 ls ${db_uri}
    aws s3 ls ${centroids_uri}
    aws s3 ls ${parts_uri}
    aws s3 ls ${index_uri}
    aws s3 ls ${sizes_uri}
    aws s3 ls ${ids_uri}
    aws s3 ls ${query_uri}
    aws s3 ls ${groundtruth_uri}
}

function verify_gp3 () {
    if [[ ! -d ${db_uri} ]]; then
	echo "${db_uri} does not exist"
    fi
    if [[ ! -d ${centroids_uri} ]]; then
	echo "${centoids_uri} does not exist"
    fi
    if [[ ! -d ${parts_uri} ]]; then
	echo "${parts_uri} does not exist"
    fi
    if [[ ! -d ${index_uri} ]]; then
	echo "${index_uri} does not exist"
    fi
    if [[ ! -d ${sizes_uri} ]]; then
	echo "${sizes_uri} does not exist"
    fi
    if [[ ! -d ${ids_uri} ]]; then
	echo "${ids_uri} does not exist"
    fi
    if [[ ! -d ${query_uri} ]]; then
	echo "${query_uri} does not exist"
    fi
    if [[ ! -d ${groundtruth_uri} ]]; then
	echo "${groundtruth_uri} does not exist"
    fi
}

function verify_all_s3 () {
    init_1M_s3
    verify_s3
    init_10M_s3
    verify_s3
    init_100M_s3
    verify_s3
    init_1B_s3
    verify_s3
    init_sift_s3
    verify_s3
}

function verify_all_gp3 () {
    init_1M_gp3
    verify_gp3
    init_10M_gp3
    verify_gp3
    init_100M_gp3
    verify_gp3
    init_1B_gp3
    verify_gp3
    init_sift_gp3
    verify_gp3
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
		local ivf_query=${2}
		shift 2
		;;
	    -h|--help)
		shift 1
		;;
	    -d|--debug)
		local _debug="-d"
		shift 1
		;;
	    -v|--verbose)
		local _verbose="-v"
		shift 1
		;;
	    --k|--knn|--k_nn)
		local _k_nn="--k ${2}"
		shift 2
		;;
	    --nqueries)
		local _nqueries="--nqueries ${2}"
		shift 2
		;;
	    --nthreads)
		local _nthreads="--nthreads ${2}"
		shift 2
		;;
	    --cluster|--nprobe)
		local _cluster="--nprobe ${2}"
		shift 2
		;;
	    --block|--blocksize)
		local _blocksize="--blocksize ${2}"
		shift 2
		;;
	    --finite)
		local _finite="--finite"
		shift 1
		;;
	    --log)
		local _log="--log ${2}"
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
	return 255
    fi

    # --index_uri ${index_uri} \
	local query="\
${ivf_query} \
--centroids_uri ${centroids_uri} \
--parts_uri ${parts_uri} \
--sizes_uri ${sizes_uri} \
--ids_uri ${ids_uri} \
--query_uri ${query_uri} \
--groundtruth_uri ${groundtruth_uri} \
${_k_nn} \
${_nqueries} \
${_nthreads} \
${_cluster} \
${_blocksize} \
${_finite} \
${_log} \
${_verbose} \
${_debug}"

	printf "================================================================\n"
	printf "=\n=\n"
	printf "${query}\n"
	eval "${query}"
}

function flat_query() {

    while [ "$#" -gt 0 ]; do
	case "$1" in
	    -x|--exec)
		local flat_query=${2}
		shift 2
		;;
	    -a|--alg|--algorithm)
		local _algorithm="--alg ${2}"
		shift 2
		;;
	    -h|--help)
		shift 1
		;;
	    -d|--debug)
		local _debug="-d"
		shift 1
		;;
	    -v|--verbose)
		local _verbose="-v"
		shift 1
		;;
	    --nqueries)
		local _nqueries="--nqueries ${2}"
		shift 2
		;;
	    --nthreads)
		local _nthreads="--nthreads ${2}"
		shift 2
		;;
	    --block|--blocksize)
		local _blocksize="--blocksize ${2}"
		shift 2
		;;
	    --nth)
		local _nth="--nth"
		shift 1
		;;
	    --log)
		local _log="--log ${2}"
		shift 2
		;;
	    -V|--validate)
		local _validate="--validate"
		shift 1
		;;
	    *)
		echo "Unknown option: $1"
		return 1
		;;
	esac
    done

    if [ -z "${flat_query}" ];
    then
	echo "flat_query executable not set"
	return 255
    fi

    local query="\
${flat_query} \
--db_uri ${db_uri} \
--query_uri ${query_uri} \
--groundtruth_uri ${groundtruth_uri} \
${_algorithm} \
${_nqueries} \
${_nthreads} \
${_blocksize} \
${_nth} \
${_log} \
${_validate} \
${_verbose} \
${_debug}"

    printf "================================================================\n"
    printf "=\n=\n"
    printf "${query}\n"
    eval "${query}"
}
