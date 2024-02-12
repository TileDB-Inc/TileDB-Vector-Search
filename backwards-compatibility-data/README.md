### What
This folder contains test indices built using different versions of TileDB-Vector-Search. It is used to test the ability of the latest version of TileDB-Vector-Search to load and query arrays built by previous versions.

### Usage
To generate new data, run:
```bash
cd apis/python
pip install .
cd ../..
python generate_data.py my_version
```
This will build new indexes and save them to `backwards-compatibility-data/data/my_version`.

To run the backwards compability test:
```bash
cd apis/python
pip install ".[test]"
pytest test/test_backwards_compatibility.py -s
```

### Versions <= 0.0.21
In order to create indexes for all releases <= 0.0.21, we run this:
- `git checkout 0.0.21`
- `git cherry-pick 3903216`
  - This is the commit we added the backwards compatibility generation script: https://github.com/TileDB-Inc/TileDB-Vector-Search/pull/206
- `git cherry-pick 60595cb`
  - This fixes a bug where indexes would use absolute paths instead of relative paths: https://github.com/TileDB-Inc/TileDB-Vector-Search/pull/193
- `~/repo/TileDB-Vector-Search-4 conda activate TileDB-Vector-Search-4`
- `(TileDB-Vector-Search-4) ~/repo/TileDB-Vector-Search-4 cd apis/python && pip install . && cd ../..`
- `(TileDB-Vector-Search-4) ~/repo/TileDB-Vector-Search-4 python backwards-compatibility-data/generate_data.py 0.0.21`
- Then clone the repo a second time and copy `/Users/parismorgan/repo/TileDB-Vector-Search-4/backwards-compatibility-data/data/0.0.21/` over to it and check it in.