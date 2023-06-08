#!/bin/bash

. ./setup.bash

ivf_query=~/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/ivf_hack
ivf_query=/home/lums/feature-vector-prototype/experimental/cmake-build-release/libtiledbvectorsearch/src/ivf_hack


init_10M
ivf_query --nqueries 16 --nprobe 16
ivf_query --nqueries 16 --nprobe 16 --finite --blocksize 0
ivf_query --nqueries 16 --nprobe 16 --finite --blocksize 1000000
ivf_query --nqueries 16 --nprobe 16 --finite --blocksize 10000000
