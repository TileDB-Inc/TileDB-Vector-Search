# Building and Running Tests

TileDB Vector Search can be built from source for either C++ or Python.

## C++

To build for C++, run:

```bash
cmake -S ./src -B ./src/build -DCMAKE_BUILD_TYPE=Debug
cmake --build ./src/build -j3
```

Then you can run the tests:

```
cmake --build ./src/build --target check
```

Alternatively, you can setup CLion to build and run tests. Just right-click on `src/CMakeLists.txt` and select `Load CMake Project`. You can then choose a target, e.g. `check`, to build and run.

## Python

To build for Python, run:

```bash
pip install .
```

You can run unit tests with `pytest`. You'll also need to install the test dependencies:

```bash
pip install ".[test]"
```

Then you can run the tests:

```bash
cd apis/python
# To run all tests.
pytest
# To run a single test and display standard output and standard error.
pytest test/test_ingestion.py -s
```

To test Demo notebooks:

```bash
cd apis/python
pip install -r test/ipynb/requirements.txt
pytest --nbmake test/ipynb
```

Credentials:

- Some tests run on TileDB Cloud using your current environment variable `TILEDB_REST_TOKEN` - you will need a valid API token for the tests to pass. See [Create API Tokens](https://docs.tiledb.com/cloud/how-to/account/create-api-tokens) for for instructions on getting one.
- For continuous integration, the token is configured for the `unittest` user and all tests should pass.

# Dependencies

## Linux

There are several dependencies needed, for Ubuntu you can install via:

```
apt-get openblas-dev build-essentials cmake3
```

To build the python API after you have the dependencies, use pip:

```bash
pip install .
```

## Docker

A docker image is also provided for simplicity:

```
docker build -t tiledb/tiledb-vector-search .
```

You run the example docker image which provides the python package with:

```
docker run --rm tiledb/tiledb-vector-search
```

# Formatting

There are two ways you can format your code.

### 1. Using `clang-format`

If you just want to format C++ code and don't want to `pip install` anything, you can install [clang-format](https://clang.llvm.org/docs/ClangFormat.html) version 17 and use that directly. Install it yourself, or by running this installation script:

```bash
./scripts/install_clang_format.sh
```

Then run it:

```bash
# Check if any files require formatting changes:
./scripts/run_clang_format.sh . clang-format 0
# Run on all files and make formatting changes:
./scripts/run_clang_format.sh . clang-format 1
```

### 2. Using `pre-commit`

Alternatively, you can format all code in the repo (i.e. C++, Python, YAML, and Markdown files) with [pre-commit](https://pre-commit.com/), though it requires installing with `pip`. Install it with:

```bash
pip install ".[formatting]"
```

Then run it:

```bash
# Run on all files and make formatting changes:
pre-commit run --all-files
# Run on single file and make formatting changes:
pre-commit run --files path/to/file.py
```
