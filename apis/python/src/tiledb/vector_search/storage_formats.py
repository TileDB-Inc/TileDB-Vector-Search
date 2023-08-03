storage_formats = {
    "0.1": {
        "CENTROIDS_ARRAY_NAME": "centroids.tdb",
        "INDEX_ARRAY_NAME": "index.tdb",
        "IDS_ARRAY_NAME": "ids.tdb",
        "PARTS_ARRAY_NAME": "parts.tdb",
        "INPUT_VECTORS_ARRAY_NAME": "input_vectors",
        "PARTIAL_WRITE_ARRAY_DIR": "write_temp",
    },
    "0.2": {
        "CENTROIDS_ARRAY_NAME": "partition_centroids",
        "INDEX_ARRAY_NAME": "partition_indexes",
        "IDS_ARRAY_NAME": "shuffled_vector_ids",
        "PARTS_ARRAY_NAME": "shuffled_vectors",
        "INPUT_VECTORS_ARRAY_NAME": "input_vectors",
        "PARTIAL_WRITE_ARRAY_DIR": "temp_data",
    },
}

STORAGE_VERSION = "0.2"
