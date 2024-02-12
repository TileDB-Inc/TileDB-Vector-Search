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
