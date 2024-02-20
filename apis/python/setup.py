import os

from skbuild import setup  # This line replaces 'from setuptools import setup'


def get_cmake_overrides():
    conf = list()

    tiledb_dir = os.environ.get("TILEDB_DIR", None)
    if tiledb_dir:
        cmake_args.append(f"-DTileDB_DIR={tiledb_dir}")

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
    val = os.environ.get(key, default="OFF")
    conf.append("-DBUILD_TESTS={}".format(val))

    key = "USE_MKL_CBLAS"
    val = os.environ.get(key, default=None)
    if val:
        conf.append("-DUSE_MKL_CBLAS={}".format(val))

    conf.append("-DTileDB_DIR=/Users/lums/Contrib/dist")

    try:
        # Make sure we use pybind11 from this python environment if available,
        # required for windows wheels due to:
        #   https://github.com/pybind/pybind11/issues/3445
        import pybind11

        pb11_path = pybind11.get_cmake_dir()
        conf.append(f"-Dpybind11_DIR={pb11_path}")
    except ImportError:
        pass

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
    use_scm_version={"root": "../../", "relative_to": __file__},
)
