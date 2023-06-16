#!/bin/bash

. ./setup.bash

ivf_query=~/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/ivf_hack
ivf_query=/home/lums/feature-vector-prototype/experimental/cmake-build-release/libtiledbvectorsearch/src/ivf_hack

arch
nproc
head -1 /proc/meminfo

cat $0



init_1B_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite

    done
done

init_1B_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 1000000

    done
done

init_1B_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 10000000

    done
done

init_1B_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 100000000

    done
done


init_100M_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1 10 100 ;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite

    done
done

init_100M_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1 10 100 10000;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 1000000

    done
done

init_100M_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1 10 100 10000;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 10000000

    done
done


init_1B_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1000 10000;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite

    done
done

init_1B_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1000 10000;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 1000000

    done
done

init_1B_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1000 10000;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 10000000

    done
done

init_1B_gp3
echo "     -|- qv_heap queries  nprobe       k    cpus          ms         qps  recall"
for nqueries in 1000 10000;
do
    for nprobe in 1 2 4 8 16 32 64 128 ;
    do

	ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize 100000000

    done
done
