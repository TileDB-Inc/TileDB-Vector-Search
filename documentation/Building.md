# Building From Source

TileDB Vector Search can be built from source. For information on dependencies, see below.

## Installation from github with pip:
```
pip install git+https://github.com/TileDB-Inc/TileDB-Vector-Search.git#subdirectory=apis/python
```

## Installation from a local checkout:
```bash
cd apis/python
pip install .
```

# Testing

You can run unit tests with `pytest`. You'll also need to install the test dependencies:
```bash
cd apis/python
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
* Some tests run on TileDB Cloud using your current environment variable `TILEDB_REST_TOKEN` - you will need a valid API token for the tests to pass. See [Create API Tokens](https://docs.tiledb.com/cloud/how-to/account/create-api-tokens) for for instructions on getting one.
* For continuous integration, the token is configured for the `unittest` user and all tests should pass.

# Dependencies

## Linux

There are several dependencies needed, for Ubuntu you can install via:
```
apt-get openblas-dev build-essentials cmake3
```

To build the python API after you have the dependencies, use pip:
```bash
cd apis/python
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

If you just want to format C++ code and don't want to `pip install` anything, you can install [clang-format](https://clang.llvm.org/docs/ClangFormat.html) version 17 and use that directly. Install it yourself, or by running this installation script:
```bash
./scripts/install_clang_format.sh
```
Then check if any files require formatting changes with:
```bash
./scripts/run_clang_format.sh . clang-format 0
```
Then make these formatting changes with:
```bash
./scripts/run_clang_format.sh . clang-format 1
```

Alternatively, you can format all code in the repo (i.e. C++, Python, YAML, and Markdown files) with [pre-commit](https://pre-commit.com/), though it requires installing with `pip`. Install it with:
```bash
cd apis/python
pip install ".[formatting]"
```
Then run it (it will autofix any issues it can) with:
```bash
# To run on all files:
pre-commit run --all-files
# To run on a single file:
pre-commit run --files path/to/file.py
```