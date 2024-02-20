<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/dev/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

# TileDB Vector Search

_TileDB-Vector-Search_ is a C++ library and Python API for vector search built on top of the [TileDB Storage Engine](https://github.com/TileDB-Inc/TileDB).

Please see the following blog posts for background:

- [Why TileDB as a Vector Database](https://tiledb.com/blog/why-tiledb-as-a-vector-database)
- [TileDB Vector Search 101](https://tiledb.com/blog/tiledb-101-vector-search/)

We have released a [LangChain integration](https://python.langchain.com/docs/integrations/vectorstores/tiledb), with others to come soon.

# Quick Links

- [Build Instructions](https://tiledb-inc.github.io/TileDB-Vector-Search/documentation/Building.html)
- [Documentation](https://tiledb-inc.github.io/TileDB-Vector-Search/)
- [Python API reference](https://tiledb-inc.github.io/TileDB-Vector-Search/documentation/reference/)

# Quick Installation

Pre-built packages are available from [PyPI](https://pypi.org/project/tiledb-vector-search) using pip:

```
pip install tiledb-vector-search
```

Or from the [tiledb conda channel](https://anaconda.org/tiledb/tiledb-vector-search) using
[conda](https://conda.io/docs/) or [mamba](https://github.com/mamba-org/mamba#installation):

```
conda install -c tiledb -c conda-forge tiledb-vector-search
```

# Contributing

We welcome contributions. Please see [`Building`](./documentation/Building.md) for
development-build instructions. For large new
features, please open an issue to discuss goals and approach in order
to ensure a smooth PR integration and review process. All contributions
must be licensed under the repository's [MIT License](./LICENSE).
