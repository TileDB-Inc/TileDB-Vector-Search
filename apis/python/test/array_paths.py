# Description: Contains paths to test arrays

import os

# TODO Use python Pathlib?
# TODO Get absolute path to project root from cmake / setup / pytest

# Use path relative to test file rather than to where pytest is invoked
this_file_path = os.path.join(os.path.dirname(__file__))
vector_search_root = this_file_path + "/../../../"

test_data_root = vector_search_root + "external/test_data/"
test_array_root = test_data_root + "arrays/"
test_file_root = test_data_root + "files/"

siftsmall_file_root = test_file_root + "siftsmall/"
siftsmall_inputs_file = siftsmall_file_root + "input_vectors.fvecs"
siftsmall_query_file = siftsmall_file_root + "queries.fvecs"
siftsmall_groundtruth_file = siftsmall_file_root + "groundtruth.ivecs"

siftsmall_root = test_array_root + "siftsmall/"
siftsmall_group_uri = siftsmall_root + "group"
siftsmall_inputs_uri = siftsmall_root + "input_vectors"
siftsmall_centroids_uri = siftsmall_root + "partition_centroids"
siftsmall_index_uri = siftsmall_root + "partition_indexes"
siftsmall_ids_uri = siftsmall_root + "shuffled_vector_ids"
siftsmall_parts_uri = siftsmall_root + "shuffled_vectors"
siftsmall_query_uri = siftsmall_root + "queries"
siftsmall_groundtruth_uri = siftsmall_root + "groundtruth"
siftsmall_dimensions = 128

siftsmall_uint8_root = test_array_root + "siftsmall_uint8/"
siftsmall_uint8_inputs_uri = siftsmall_uint8_root + "input_vectors"
siftsmall_uint8_ids_uri = siftsmall_uint8_root + "shuffled_vector_ids"
siftsmall_uint8_groundtruth_uri = siftsmall_uint8_root + "groundtruth"

bigann10k_root = test_array_root + "bigann10k/"
bigann10k_inputs_uri = bigann10k_root + "input_vectors"
bigann10k_ids_uri = bigann10k_root + "shuffled_vector_ids"
bigann10k_groundtruth_uri = bigann10k_root + "groundtruth"

fmnistsmall_root = test_array_root + "fmnistsmall/"
fmnistsmall_inputs_uri = fmnistsmall_root + "input_vectors"
fmnistsmall_ids_uri = fmnistsmall_root + "shuffled_vector_ids"
fmnistsmall_groundtruth_uri = fmnistsmall_root + "groundtruth"
