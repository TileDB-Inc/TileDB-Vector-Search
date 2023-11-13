import tiledb

storage_formats = {
    "0.1": {
        "CENTROIDS_ARRAY_NAME": "centroids.tdb",
        "INDEX_ARRAY_NAME": "index.tdb",
        "IDS_ARRAY_NAME": "ids.tdb",
        "PARTS_ARRAY_NAME": "parts.tdb",
        "INPUT_VECTORS_ARRAY_NAME": "input_vectors",
        "EXTERNAL_IDS_ARRAY_NAME": "external_ids",
        "PARTIAL_WRITE_ARRAY_DIR": "write_temp",
        "DEFAULT_ATTR_FILTERS": None,
        "UPDATES_ARRAY_NAME": "updates",
        "SUPPORT_TIMETRAVEL": False,
    },
    "0.2": {
        "CENTROIDS_ARRAY_NAME": "partition_centroids",
        "INDEX_ARRAY_NAME": "partition_indexes",
        "IDS_ARRAY_NAME": "shuffled_vector_ids",
        "PARTS_ARRAY_NAME": "shuffled_vectors",
        "INPUT_VECTORS_ARRAY_NAME": "input_vectors",
        "EXTERNAL_IDS_ARRAY_NAME": "external_ids",
        "PARTIAL_WRITE_ARRAY_DIR": "temp_data",
        "DEFAULT_ATTR_FILTERS": tiledb.FilterList([tiledb.ZstdFilter()]),
        "UPDATES_ARRAY_NAME": "updates",
        "SUPPORT_TIMETRAVEL": False,
    },
    "0.3": {
        "CENTROIDS_ARRAY_NAME": "partition_centroids",
        "INDEX_ARRAY_NAME": "partition_indexes",
        "IDS_ARRAY_NAME": "shuffled_vector_ids",
        "PARTS_ARRAY_NAME": "shuffled_vectors",
        "INPUT_VECTORS_ARRAY_NAME": "input_vectors",
        "EXTERNAL_IDS_ARRAY_NAME": "external_ids",
        "PARTIAL_WRITE_ARRAY_DIR": "temp_data",
        "DEFAULT_ATTR_FILTERS": tiledb.FilterList([tiledb.ZstdFilter()]),
        "UPDATES_ARRAY_NAME": "updates",
        "SUPPORT_TIMETRAVEL": True,
    },
}

STORAGE_VERSION = "0.3"
