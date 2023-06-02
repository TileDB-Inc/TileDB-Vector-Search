# Prerequisites

1. Create and activate a Python environment with pybind11 available

  a. Download minimamba:
  b. `mamba env create -f environment.yml`
  c. `mamba activate tiledbvspy`


# Build Instructions

## Build and install with pip

(TODO: DOES NOT WORK YET)

```
CMAKE_ARGS="-DTileDB_DIR=/Users/inorton/work/bld/TileDB-2.15/rel/dist/lib/cmake/TileDB" pip install -e
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

# ====

X. Usage example

```
In [1]: import tiledbvspy

In [2]: v = tiledbvspy.get_v()

In [3]: v_np = np.array(v)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Input In [3], in <cell line: 1>()
----> 1 v_np = np.array(v)

NameError: name 'np' is not defined

In [4]: import tiledbvspy, numpy as np

In [5]: v_np = np.array(v)

In [6]: v_np
Out[6]: array([1., 2., 3., 4., 5., 6., 7.], dtype=float32)

In [7]: v_np[1] = 10

In [8]: v
Out[8]: <tiledbvspy.Vector_f32 at 0x103030db0>

In [9]: v[1]
Out[9]: 2.0

In [10]: v[0]
Out[10]: 1.0

In [11]: v[2]
Out[11]: 3.0

In [12]: v[3]
Out[12]: 4.0

In [13]: v_np_view = np.array(v, copy=False)

In [14]: v_np_view
Out[14]: array([1., 2., 3., 4., 5., 6., 7.], dtype=float32)

In [15]: v_np_view[2] = 12.5

In [16]: v_np_view
Out[16]: array([ 1. ,  2. , 12.5,  4. ,  5. ,  6. ,  7. ], dtype=float32)

In [17]: v[2]
Out[17]: 12.5 # original data modified
```
