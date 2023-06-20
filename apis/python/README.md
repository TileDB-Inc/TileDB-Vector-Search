# Prerequisites

1. Create and activate a Python environment with pybind11 available

  a. Download minimamba:
  b. `mamba env create -f environment.yml`
  c. `mamba activate tiledbvspy`


# Build Instructions

## Build and install with pip

See `setup.py` for supported configuration overrides, including the libtiledb path.

```
pip install -e .
```

## Direct build, then export PYTHONPATH

For development purposes, at the moment it is simpler to build directly and export the
extension directory into PYTHONPATH.

1. Run the top level cmake build with `-DTILEDB_VS_PYTHON=ON`

```
mkdir fvp-build
cd fvp-build
cmake -DTILEDB_VS_PYTHON=ON ~/work/git/feature-vector-type
<cmake --build . or make -j4 etc.>
```


2. Look at `fvp-build/python`. There should be a Python extension module:

```
pydev ❯ pwd
/Users/inorton/work/bld/fvp-build/python

~/work/bld/fvp-build/python
pydev ❯ ls
CMakeFiles                       cmake_install.cmake              tiledbvspy.cpython-310-darwin.so
```

3. Export the extension path as PYTHONPATH

```
export PYTHONPATH= ~/work/bld/fvp-build/python
```

3. Run Python, and `import tiledbvspy`

```
import tiledbvspy
```