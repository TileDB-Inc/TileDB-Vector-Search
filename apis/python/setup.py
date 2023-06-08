from skbuild import setup  # This line replaces 'from setuptools import setup'

import os

setup(
    name="tiledb-vector-search",
    description="Vector Search with TileDB",
    author='TileDB',
    license="MIT",
    python_requires="~=3.7",
    packages=['tiledb.vector_search'],
    cmake_source_dir="../../experimental",
    cmake_args=['-DTILEDB_VS_PYTHON=ON'],
)
