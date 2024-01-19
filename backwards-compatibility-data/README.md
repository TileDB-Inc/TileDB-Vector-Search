### What
This folder contains test indices built using different versions of TileDB-Vector-Search. It is used to test the ability of the latest version of TileDB-Vector-Search to load and query arrays built by previous versions.

### Usage
To generate new data, run:
- `python generate_data.py x.x.x`
This will create a new folder in the `data` directory with the version. This folder will contain the arrays built by the current version of TileDB-Vector-Search.

To run a backwards compability test, run:
- `cd ~/repo/TileDB-Vector-Search && pytest apis/python/test/test_backwards_compatibility.py -s`