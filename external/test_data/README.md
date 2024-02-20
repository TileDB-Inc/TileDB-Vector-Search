## Test Data

This directory contains test data for TileDB-Vector-Search unit tests.

The test data is organized into two subdirectories, containing raw data files (`files`) and
TileDB arrays (`arrays`)

### Basic Test Data

The TileDB-Vector-Search repo contains small data sets for testing.
When you first checkout TileDB-Vector-Search from github you will see the following:

```text
├── arrays
│   └── siftsmall
│   └── bigann10k
└── files
    ├── diskann
    └── siftsmall
```

Within each subdirectory are an input dataset for vector search, a query dataset, and a groundtruth dataset for the queries in the query dataset.

```text
├── arrays
│   ├── bigann10k
│   │   ├── input_vectors
│   │   ├── queries
│   │   └── groundtruth
│   └── siftsmall
│       ├── input_vectors
│       ├── queries
│       └── groundtruth
└── files
    └── siftsmall
        ├── input_vectors.fvecs
        ├── queries.fvecs
        └── groundtruth.ivecs
```

### IVF Indexes

The array test data subdirectory additionally includes arrays comprising the IVF index for the vector data.

```text
└── arrays
    ├── bigann10k
    │   ├── input_vectors
    │   ├── queries
    │   ├── groundtruth
    │   ├── partition_centroids
    │   ├── partition_indexes
    │   ├── shuffled_vectors
    │   └── shuffled_vector_ids
    └── siftsmall
        ├── input_vectors
        ├── queries
        ├── groundtruth
        ├── partition_centroids
        ├── partition_indexes
        ├── shuffled_vectors
        └── shuffled_vector_ids
```

### Additional Test Data

Additional test arrays (e.g., for benchmarking or additional testing) can be downloaded:

- `bigann1M`
- `sift`

To download these files:

```bash
# TBD
```
