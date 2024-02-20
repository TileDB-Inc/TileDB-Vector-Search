#!/bin/bash

# ftype = float32
# idtype = uint32
ftype=float
idtype=uint32

# ftype = float32
# idtype = uint64
ftype=float
idtype=uint64

# ftype = uint8
# idtype = uint32
ftype=uint8
idtype=uint32

# ftype = uint8
# idtype = uint64
ftype=uint8
idtype=uint64

# default (ftype = uint8, idtype = uint64)

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/index --ftype float --idtype uint32
./libtiledbvectorsearch/src/vamana/index --ftype float --idtype uint32 \
--db_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_learn \
--index_uri /tmp/index_siftsmall_learn_R16_L16_A1.2 \
-R 16 -L 16 --alpha 1.2 --nthreads 1 -v -d --log - --force

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --ftype float --idtype uint32
./libtiledbvectorsearch/src/vamana/query --ftype float --idtype uint32 \
--index_uri /tmp/index_siftsmall_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_query \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_groundtruth \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/index --ftype float --idtype uint64
./libtiledbvectorsearch/src/vamana/index --ftype float --idtype uint64 \
--db_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_learn \
--index_uri /tmp/index_siftsmall_learn_R16_L16_A1.2 \
-R 16 -L 16 --alpha 1.2 --nthreads 1 -v -d --log - --force

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --ftype float --idtype uint64
./libtiledbvectorsearch/src/vamana/query --ftype float --idtype uint64 \
--index_uri /tmp/index_siftsmall_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_query \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_groundtruth \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --ftype float
./libtiledbvectorsearch/src/vamana/query --ftype float \
--index_uri /tmp/index_siftsmall_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_query \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_groundtruth \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/index --ftype float
./libtiledbvectorsearch/src/vamana/index --ftype float \
--db_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_learn \
--index_uri /tmp/index_siftsmall_learn_R16_L16_A1.2 \
-R 16 -L 16 --alpha 1.2 --nthreads 1 -v -d --log - --force

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --ftype float --idtype uint64
./libtiledbvectorsearch/src/vamana/query --ftype float --idtype uint64 \
--index_uri /tmp/index_siftsmall_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_query \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/siftsmall/siftsmall_groundtruth \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/index --ftype uint8 --idtype uint32
./libtiledbvectorsearch/src/vamana/index --ftype uint8 --idtype uint32 \
--db_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann1M_base \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
-R 16 -L 16 --alpha 1.2 --nthreads 1 -v -d --log - --force

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --ftype uint8 --idtype uint32
./libtiledbvectorsearch/src/vamana/query --ftype uint8 --idtype uint32 \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/query_public_10k \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann_1M_GT_nnids \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/index --ftype uint8 --idtype uint32
./libtiledbvectorsearch/src/vamana/index --ftype uint8 --idtype uint32 \
--db_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann1M_base \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
-R 16 -L 16 --alpha 1.2 --nthreads 1 -v -d --log - --force

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --idtype uint32
./libtiledbvectorsearch/src/vamana/query --idtype uint32 \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/query_public_10k \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann_1M_GT_nnids \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/index --idtype uint32
./libtiledbvectorsearch/src/vamana/index --idtype uint32 \
--db_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann1M_base \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
-R 16 -L 16 --alpha 1.2 --nthreads 1 -v -d --log - --force

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --ftype uint8 --idtype uint32
./libtiledbvectorsearch/src/vamana/query --ftype uint8 --idtype uint32 \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/query_public_10k \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann_1M_GT_nnids \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/index --ftype uint8 --idtype uint64
./libtiledbvectorsearch/src/vamana/index --ftype uint8 --idtype uint64 \
--db_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann1M_base \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
-R 16 -L 16 --alpha 1.2 --nthreads 1 -v -d --log - --force

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --ftype uint8 --idtype uint64
./libtiledbvectorsearch/src/vamana/query --ftype uint8 --idtype uint64 \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/query_public_10k \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann_1M_GT_nnids \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

echo ./libtiledbvectorsearch/src/vamana/query --ftype uint8
./libtiledbvectorsearch/src/vamana/query --ftype uint8 \
--index_uri /tmp/index_bigann1M_learn_R16_L16_A1.2 \
--query_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/query_public_10k \
--groundtruth_uri /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann_1M_GT_nnids \
-k 10 -L 10 --nthreads 1 -v -d --log -

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
