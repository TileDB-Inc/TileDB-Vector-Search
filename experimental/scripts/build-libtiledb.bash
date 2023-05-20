#!/bin/bash


# for libtiledb
#    ../bootstrap --enable-s3 --force-build-all-deps --disable-werror --disable-tests

# pd ~/TileDB/TileDB
mkdir -p ~/Contrib/libtiledb
pushd ~/Contrib/libtiledb
git clone git@github.com/TileDB-Inc/TileDB
cd TileDB
git checkout 2.15.2
/bin/rm -rf cmake-build-release
mkdir -p cmake-build-release
cd cmake-build-release
../bootstrap --enable-s3 --force-build-all-deps --disable-werror --disable-tests --prefix=/Users/lums/Contrib/dist
make
make -C tiledb install
popd
