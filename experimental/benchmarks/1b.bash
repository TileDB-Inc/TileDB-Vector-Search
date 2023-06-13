#!/bin/bash

. ./setup.bash

ivf_query=~/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/ivf_hack
ivf_query=/home/lums/feature-vector-prototype/experimental/cmake-build-release/libtiledbvectorsearch/src/ivf_hack


init_1B_gp3
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe}

    done
done


init_1B_nvme
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe}

    done
done

init_1B_nvme
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite

    done
done


init_1B_nvme
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 10000000

    done
done


init_1B_nvme
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 5000

    done
done


init_1B_s3
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe}

    done
done

init_1B_s3
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite

    done
done


init_1B_s3
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 10000000

    done
done


init_1B_s3
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 5000

    done
done


