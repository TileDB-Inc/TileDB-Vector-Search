# Running big-ann-benchmarks

We have implemented a [big-ann-benchmarks](https://big-ann-benchmarks.com) runner for TileDB-Vector-Search,
which is available in the `tiledb` branch of our fork:
- [https://github.com/TileDB-Inc/big-ann-benchmarks/tree/tiledb](https://github.com/TileDB-Inc/big-ann-benchmarks/tree/tiledb)

## Building

1) Build the `Dockerfile` at the root of this repository:

```
cd tiledb-vector-search
docker build -f Dockerfile . -t tiledb_vs
```

2) Build the TileDB docker image in the big-ann fork (requires image from step 1):

```
git clone https://github.com/TileDB-Inc/big-ann-benchmarks/tree/tiledb
cd big-ann-benchmarks
docker build -f install/Dockerfile.tiledb .
```
