import tiledb

storage_formats = {
    "0.1": {
        "CENTROIDS_ARRAY_NAME": "centroids.tdb",
        "INDEX_ARRAY_NAME": "index.tdb",
        "IDS_ARRAY_NAME": "ids.tdb",
        "PARTS_ARRAY_NAME": "parts.tdb",
        "INPUT_VECTORS_ARRAY_NAME": "input_vectors",
        "TRAINING_INPUT_VECTORS_ARRAY_NAME": "training_input_vectors",
        "EXTERNAL_IDS_ARRAY_NAME": "external_ids",
        "PARTIAL_WRITE_ARRAY_DIR": "write_temp",
        "DEFAULT_ATTR_FILTERS": None,
        "UPDATES_ARRAY_NAME": "updates",
        "OBJECT_METADATA_ARRAY_NAME": "object_metadata",
        "SUPPORT_TIMETRAVEL": False,
        "FEATURE_VECTORS_ARRAY_NAME": "feature_vectors",
        "ADJACENCY_SCORES_ARRAY_NAME": "adjacency_scores",
        "ADJACENCY_IDS_ARRAY_NAME": "adjacency_ids",
        "ADJACENCY_ROW_index_ARRAY_NAME": "adjacency_row_index",
    },
    "0.2": {
        
        # Input data
        "INPUT_VECTORS_ARRAY_NAME": "input_vectors", # = [[1, 2, 3, 4], [5, 6, 7, 8]]
        "EXTERNAL_IDS_ARRAY_NAME": "external_ids", # = [99, 100]
        "TRAINING_INPUT_VECTORS_ARRAY_NAME": "training_input_vectors",
        
        # Vectors that may be shuffled.
        "PARTS_ARRAY_NAME": "shuffled_vectors", #= [[5, 6, 7, 8], [1, 2, 3, 4]]
        # Ids for the vectors.
        "IDS_ARRAY_NAME": "shuffled_vector_ids", # = [100, 99]
        
        # Centroids
        "CENTROIDS_ARRAY_NAME": "partition_centroids",
        # Mapping from the centroids to the shuffled vector indexes
        "INDEX_ARRAY_NAME": "partition_indexes",
        
        "PARTIAL_WRITE_ARRAY_DIR": "temp_data",
        "DEFAULT_ATTR_FILTERS": tiledb.FilterList([tiledb.ZstdFilter()]),
        "UPDATES_ARRAY_NAME": "updates",
        "OBJECT_METADATA_ARRAY_NAME": "object_metadata",
        "SUPPORT_TIMETRAVEL": False,
        
        "FEATURE_VECTORS_ARRAY_NAME": "feature_vectors",
        "ADJACENCY_SCORES_ARRAY_NAME": "adjacency_scores",
        "ADJACENCY_IDS_ARRAY_NAME": "adjacency_ids",
        "ADJACENCY_ROW_index_ARRAY_NAME": "adjacency_row_index",
    },
# Main input is the vectors, and for each vector we have an external id.
# Flat
    # no re-ordering. so whatever we get in the input we put in the output.



    "0.3": {
        "CENTROIDS_ARRAY_NAME": "partition_centroids",
        #  partition number vectors with each ha
        "INDEX_ARRAY_NAME": "partition_indexes",
        # indexes in the shuffled vector array - shuffled_vectors is a dense array. we shuffle the vectors but oyu don't know the boundary.
        # for partition 4 read from position 0 to 9, for partition 5 read from position 10 to 19, etc
        # this is a 1d array
        # dimension = number of partitions. each has two attributes: start and end
        "IDS_ARRAY_NAME": "shuffled_vector_ids",
            # these are the ids for each vector in shuffled_vectors
        "PARTS_ARRAY_NAME": "shuffled_vectors",
            # for flat this is just the input vectors
            # for ivf we do shuffle
        "INPUT_VECTORS_ARRAY_NAME": "input_vectors",
            # [optional for ingestion] this is just the input vectors provided
        "TRAINING_INPUT_VECTORS_ARRAY_NAME": "training_input_vectors",
            # [optional for ingestion] this is just part of training the centroids during ingestion
        "EXTERNAL_IDS_ARRAY_NAME": "external_ids",
            # 1 2 3 4
            # these are the ids that the user provides us
        "PARTIAL_WRITE_ARRAY_DIR": "temp_data",
        "DEFAULT_ATTR_FILTERS": tiledb.FilterList([tiledb.ZstdFilter()]),
        "UPDATES_ARRAY_NAME": "updates",
        "OBJECT_METADATA_ARRAY_NAME": "object_metadata",
        "SUPPORT_TIMETRAVEL": True,
        
        "FEATURE_VECTORS_ARRAY_NAME": "feature_vectors",
        "ADJACENCY_SCORES_ARRAY_NAME": "adjacency_scores",
        "ADJACENCY_IDS_ARRAY_NAME": "adjacency_ids",
        "ADJACENCY_ROW_index_ARRAY_NAME": "adjacency_row_index",
    },
}

STORAGE_VERSION = "0.3"


def validate_storage_version(storage_version):
    if storage_version not in storage_formats:
        valid_versions = ", ".join(storage_formats.keys())
        raise ValueError(
            f"Invalid storage version: {storage_version} - valid versions are [{valid_versions}]"
        )
