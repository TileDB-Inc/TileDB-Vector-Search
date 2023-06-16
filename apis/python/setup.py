from skbuild import setup  # This line replaces 'from setuptools import setup'

import os

cmake_args = ['-DTILEDB_VS_PYTHON=ON']
if tiledb_dir := os.environ.get("TILEDB_DIR", None):
    cmake_args.append(f"-DTileDB_DIR={tiledb_dir}")


setup(
    name="tiledb-vector-search",
    description="Vector Search with TileDB",
    author="TileDB",
    python_requires="~=3.7",
    packages=["tiledb.vector_search"],
    package_dir={"": "src"},
    cmake_source_dir="../../src",
    cmake_args=["-DTILEDB_VS_PYTHON=ON"],
    cmake_args=cmake_args,
    cmake_install_target="install-libtiledbvectorsearch",
    cmake_install_dir="src/tiledb/vector_search",
)
