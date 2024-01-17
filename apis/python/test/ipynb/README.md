# Cloud Notebook tests

Run all tests:

```
pip install -r test/ipynb/requirements.txt
pytest --nbmake test/ipynb
```

This is using [nbmake](https://github.com/treebeardtech/nbmake) to test notebooks.
Credentials:

- Local tests: use `TILEDB_REST_TOKEN` -- you will need a valid API token for the tests to pass
- For continuous integration, the token is configured for the `unittest` user and all tests should pass

When changes get merged in these files make sure that you also propagate the changes to the Cloud registered notebooks:

- [tiledb_101_vector_search.ipynb](https://cloud.tiledb.com/notebooks/details/TileDB-Inc/b05dd4b4-ba1c-41c6-a3c9-a1de70abb039/preview)
- [staging-vector-search-checks-py.ipynb](https://console.dev.tiledb.io/notebooks/details/TileDB-Inc/299dd052-6b45-4943-88ae-37639b7b4b48/preview)
- [image-search-dashboard](https://cloud.tiledb.com/notebooks/details/TileDB-Inc/9289229a-1742-4f99-9b86-0bb1339b31a0/preview)
