#!/bin/bash

. ./setup.bash

ivf_query=~/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/ivf_hack
ivf_query=/home/lums/feature-vector-prototype/experimental/cmake-build-release/libtiledbvectorsearch/src/ivf_hack


init_100M_gp3
for nprobe in 1 2 4 8 16 32 64 128 256;
do
    for nqueries in 1 10 100 1000 5000 10000;
    do
	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --blocksize 10000000
    done
done


init_1B_gp3
for nprobe in 1 2 4 8 16 32 64 128 256;
do
    for nqueries in 1 10 100 1000 5000 10000;
    do
	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --blocksize 10000000
    done
done

