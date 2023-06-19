# Building From Source

TileDB Vector Search can be built from source. 

## Linux

There are several dependencies needed, for Ubuntu you can install via:
```
apt-get openblas-dev build-essentials cmake3 
```

To build the python API after you have the dependencies, use pip:
```
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