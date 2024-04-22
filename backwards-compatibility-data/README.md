### What

This folder contains test indices built using different versions of TileDB-Vector-Search. It is used to test the ability of the latest version of TileDB-Vector-Search to load and query arrays built by previous versions.

In CI we run `generate_data.py` on each release and on major and minor version bump releases create PR with the generated data into `main`. We do not check in the generated data for patch releases.

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
- Fix any merge conflicts.
- Manually check for any instances of `group.add(` which are not using `add_to_group()`.
- `~/repo/TileDB-Vector-Search-4 conda activate TileDB-Vector-Search-4`
- `(TileDB-Vector-Search-4) ~/repo/TileDB-Vector-Search-4 cd apis/python && pip install . && cd ../..`
- `(TileDB-Vector-Search-4) ~/repo/TileDB-Vector-Search-4 python backwards-compatibility-data/generate_data.py 0.0.21`
- Copy `/Users/parismorgan/repo/TileDB-Vector-Search-4/backwards-compatibility-data/data/0.0.21/` somewhere and then check it in.

We currently have the following old indexes created:

- `0.0.10` (using storage format 0.1)
  - Here we generate data for `STORAGE_VERSION` `0.1`. There is no API to do this, so we manually set `STORAGE_VERSION` in `storage_formats.py` before creating the index.
  - You can see the code used to generate it here: https://github.com/TileDB-Inc/TileDB-Vector-Search/tree/jparismorgan/backwards-compat-0.0.10
- `0.0.17` (using storage format 0.2)
  - You can see the code used to generate it here: https://github.com/TileDB-Inc/TileDB-Vector-Search/tree/jparismorgan/backwards-compat-0.0.17
- `0.0.21`
  - You can see the code used to generate it here: https://github.com/TileDB-Inc/TileDB-Vector-Search/tree/jparismorgan/backwards-compat-0.0.21-working

We choose these three because they let us test all old storage versions (old meaing before we could automatically generate backwards compatibility indexes). If you need to add more indexes, please feel free to do so.
