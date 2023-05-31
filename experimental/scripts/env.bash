#!/bin/bash

function init_paths() {

    default_region=us-east-1
    default_prefix=s3://tiledb-andrew
    default_arch=m1

    declare -g region=${default_region}
    declare -g s3_prefix=${default_prefix}
    declare -g arch=${default_arch}

    if [[ $# -gt 0 ]]; then
	declare -g region=${1}
    fi
    if [[ $# -gt 1 ]]; then
	declare -g s3_prefix=${2}
    fi
    if [[ $# -gt 2 ]]; then
	declare -g arch=${3}
    fi

    # declare -g region=us-west-2
    # declare -g s3_prefix=s3://tiledb-lums

    declare -g kmeans_prefix=${s3_prefix}/kmeans
    declare -g ivf_hack_prefix=${kmeans_prefix}/ivf_hack
    declare -g arch_prefix=${ivf_hack_prefix}/${arch}
    
    declare -g sift_prefix=${s3_prefix}/sift
    
    declare -g local_array_prefix=/Users/lums/TileDB/feature-vector-prototype/experimental/data/arrays/
    declare -g local_sift_prefix=${local_array_prefix}/sift
    declare -g local_kmeans_prefix=${local_array_prefix}/kmeans
    
    declare -g default_ivf_build_prefix=/Users/lums/TileDB/feature-vector-prototype/experimental/cmake-build-release
}

function print_schema() {
    s3_prefix=${2}
    echo ${s3_prefix} at ${1}    

python3 -c """
import tiledb
tiledb.default_ctx({\"vfs.s3.region\": \"${1}\"})

print(\"# ${arch_prefix}/ids\")
a = tiledb.open(\"${arch_prefix}/ids\")
print(a.schema)

print(\"# ${arch_prefix}/index\")
b = tiledb.open(\"${arch_prefix}/index\")
print(b.schema)

print(\"# ${arch_prefix}/parts\")
c = tiledb.open(\"${arch_prefix}/parts\")
print(c.schema)

print(\"# ${ivf_hack_prefix}/centroids\")
d = tiledb.open(\"${ivf_hack_prefix}/centroids\")
print(d.schema)

print(\"# ${sift_prefix}/sift_base\")
e = tiledb.open(\"${sift_prefix}/sift_base\")
print(e.schema)

print(\"# ${sift_prefix}/sift_query\")
f = tiledb.open(\"${sift_prefix}/sift_query\")
print(f.schema)
"""
}

function build_ivf_hack() {
    ivf_build_prefix=${default_ivf_build_prefix}
    
    if [[ $# -gt 0 ]]; then
	ivf_build_prefix=${1}
    fi
    if [[ $# -gt 1 ]]; then
	branch=${2}
    fi

    pushd ${ivf_build_prefix}
    if [ -n "${branch}" ];
    then
       git checkout ${branch}
    fi
    make clean
    make ivf_hack
    popd
}
			 


function ivf_hack() {
    ivf_build_prefix=${default_ivf_build_prefix}
    write=""
    dryrun=""
    branch="lums/tmp/part"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      -n|--dryrun)
        dryrun="--dryrun"
        shift 1
        ;;
      -w|--write)
        write="--write"
        shift 1
        ;;
      -p|--prefix)
        ivf_build_prefix=${2}
        shift 2
        ;;
      -b|--branch)
        branch=${2}
        shift 2
        ;;
      -x|--build)
	  build=1
	  shift 1
	  ;;
      *)
        echo "Unknown option: $1"
        return 1
        ;;
    esac
  done

#  echo write is ${write}
#  echo dryrun is ${dryrun}
#  echo prefix is ${ivf_build_prefix}

  
  if [ ! -d "${ivf_build_prefix}" ]; then
      echo "${ivf_build_prefix} does not exist!"
      return
  fi

  if [[ -n "${build}" ]];
  then
      build_ivf_hack ${ivf_build_prefix} ${branch}
  fi


echo RUNNING ${ivf_build_prefix}/src/ivf_hack 

${ivf_build_prefix}/src/ivf_hack \
    --id_uri ${arch_prefix}/ids --index_uri ${arch_prefix}/index --part_uri ${arch_prefix}/parts  \
    --centroids_uri ${arch_prefix}/centroids --db_uri ${sift_prefix}/sift_base                    \
    --query_uri ${sift_prefix}/sift_query --region ${region} --debug ${write} ${dryrun}
}
 
