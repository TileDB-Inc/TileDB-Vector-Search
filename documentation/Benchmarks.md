# Benchmarks

We have implemented a [big-ann-benchmarks](https://big-ann-benchmarks.com) interface for TileDB-Vector-Search,
which is available in the `tiledb` branch of our fork:

- [https://github.com/TileDB-Inc/big-ann-benchmarks/tree/tiledb](https://github.com/TileDB-Inc/big-ann-benchmarks/tree/tiledb). This interface implements two new algorithms: `tiledb-flat` and `tiledb-ivf-flat`, which are usable within the framework's runner.

## Building

1. Build the `Dockerfile` at the root of this repository:

```
cd tiledb-vector-search
docker build -f Dockerfile . -t tiledb_vs
```

2. Build the TileDB docker image in the big-ann fork (requires image from step 1):

```
git clone https://github.com/TileDB-Inc/big-ann-benchmarks/tree/tiledb
cd big-ann-benchmarks
docker build -f install/Dockerfile.tiledb . -t billion-scale-benchmark-tiledb
```

## Running benchmarks

1. Create a local dataset.

   note: the `create_dataset.py` command will download
   remote files the first time it runs, some of which can total >100GB). Use `--skip-data`
   to avoid downloading the large base set.

   _This_ command will download 7.7MB of data:

```
python create_dataset.py --dataset bigann-10M --skip-data
```

2. Run the benchmarks, choosing either `tiledb-flat` or `tiledb-ivf-flat`:

```
python run.py --dataset bigann-10M --algorithm tiledb-flat
```
