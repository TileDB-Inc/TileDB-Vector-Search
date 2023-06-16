#!/bin/bash

. ./env.bash

# These all work as of 2023-05-20 15:09

init_paths us-east-1 /Users/lums/TileDB/feature-vector-prototype/external/data/arrays
ivf_hack --write --dryrun

init_paths us-east-1 s3://tiledb-andrew
ivf_hack --write --dryrun

init_paths us-west-2 s3://tiledb-lums
ivf_hack --write --dryrun
