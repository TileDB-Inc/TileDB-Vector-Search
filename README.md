<a href="https://tiledb.com"><img src="https://github.com/TileDB-Inc/TileDB/raw/main/doc/source/_static/tiledb-logo_color_no_margin_@4x.png" alt="TileDB logo" width="400"></a>

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

# Quick Start

## Basic Vector Search

```python
import tiledb.vector_search as vs
import numpy as np

# Create an index
uri = "my_index"
vectors = np.random.rand(10000, 128).astype(np.float32)

vs.ingest(
    index_type="VAMANA",
    index_uri=uri,
    input_vectors=vectors,
    l_build=100,
    r_max_degree=64
)

# Query the index
index = vs.VamanaIndex(uri)
query = np.random.rand(128).astype(np.float32)
distances, ids = index.query(query, k=10)
```

## Filtered Vector Search

Perform nearest neighbor search restricted to vectors matching metadata criteria. This feature uses the **Filtered-Vamana** algorithm, which maintains high recall (>90%) even for highly selective filters.

```python
import tiledb.vector_search as vs
import numpy as np

# Create index with filter labels
uri = "my_filtered_index"
vectors = np.random.rand(10000, 128).astype(np.float32)

# Assign labels to vectors (e.g., by data source)
filter_labels = {
    i: [f"source_{i % 10}"]  # Each vector has a label
    for i in range(10000)
}

vs.ingest(
    index_type="VAMANA",
    index_uri=uri,
    input_vectors=vectors,
    filter_labels=filter_labels,  # Add filter labels during ingestion
    l_build=100,
    r_max_degree=64
)

# Query with filter - only return results from source_5
index = vs.VamanaIndex(uri)
query = np.random.rand(128).astype(np.float32)

distances, ids = index.query(
    query,
    k=10,
    where="source == 'source_5'"  # Filter condition
)

# Query with multiple labels using IN clause
distances, ids = index.query(
    query,
    k=10,
    where="source IN ('source_1', 'source_2', 'source_5')"
)
```

### Filtered Search Performance

Filtered search achieves **>90% recall** even for highly selective filters:

- **Specificity 10⁻³** (0.1% of data): >95% recall
- **Specificity 10⁻⁶** (0.0001% of data): >90% recall

This is achieved through the **Filtered-Vamana** algorithm, which modifies graph construction and search to preserve connectivity for rare labels. Post-filtering approaches degrade significantly at low specificity, while Filtered-Vamana maintains high recall with minimal performance overhead.

Based on: [Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters](https://doi.org/10.1145/3543507.3583552) (Gollapudi et al., WWW 2023)

# Contributing

We welcome contributions. Please see [`Building`](./documentation/Building.md) for
development-build instructions. For large new
features, please open an issue to discuss goals and approach in order
to ensure a smooth PR integration and review process. All contributions
must be licensed under the repository's [MIT License](./LICENSE).
