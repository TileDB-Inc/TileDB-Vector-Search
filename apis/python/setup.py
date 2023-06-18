from skbuild import setup  # This line replaces 'from setuptools import setup'

import os

def get_cmake_overrides():
    import sys

    conf = list()

    key = "CMAKE_OSX_DEPLOYMENT_TARGET"
    val = os.environ.get(key, default=None)
    if val:
        conf.append("-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(val))

    key = "DOWNLOAD_TILEDB_PREBUILT"
    val = os.environ.get(key, default=None)
    if val:
        conf.append("-DDOWNLOAD_TILEDB_PREBUILT={}".format(val))

    key = "CMAKE_BUILD_TYPE"
    val = os.environ.get(key, default=None)
    if val:
        conf.append("-DCMAKE_BUILD_TYPE={}".format(val))

    key = "BUILD_TESTS"
    val = os.environ.get(key, default=None)
    if val:
        conf.append("-DBUILD_TESTS={}".format(val))

    key = "USE_MKL_CBLAS"
    val = os.environ.get(key, default=None)
    if val:
        conf.append("-DUSE_MKL_CBLAS={}".format(val))

    return conf

cmake_args = ["-DTILEDB_VS_PYTHON=ON"]

cmake_args += get_cmake_overrides()

setup(
    name="tiledb-vector-search",
    description="Vector Search with TileDB",
    author="TileDB",
    python_requires="~=3.7",
    packages=["tiledb.vector_search"],
    package_dir={"": "src"},
    cmake_source_dir="../../src",
    cmake_args=cmake_args,
    cmake_install_target="install-libtiledbvectorsearch",
    cmake_install_dir="src/tiledb/vector_search",
)
