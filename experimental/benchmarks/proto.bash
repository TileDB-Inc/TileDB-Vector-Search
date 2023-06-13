#!/bin/bash

. ./setup.bash

ivf_query=~/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/ivf_hack
ivf_query=/home/lums/feature-vector-prototype/experimental/cmake-build-release/libtiledbvectorsearch/src/ivf_hack



init_1M_s3
for nprobe in 1 2 4 8 16 32 64 128 256;
do
    for nqueries in 1 10 100 1000 5000 10000;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe}

    done
done

init_10M_s3
for nprobe in 1 2 4 8 16 32 64 128 256;
do
    for nqueries in 1 10 100 1000 5000 10000;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe}

    done
done

init_100M_s3
for nprobe in 1 2 4 8 16 32 64 128 256;
do
    for nqueries in 1 10 100 1000 5000 10000;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe}

    done
done
