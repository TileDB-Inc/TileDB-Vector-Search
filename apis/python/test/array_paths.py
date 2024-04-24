# Description: Contains paths to test arrays

import os

# TODO Use python Pathlib?
# TODO Get absolute path to project root from cmake / setup / pytest

# Use path relative to test file rather than to where pytest is invoked
this_file_path = os.path.join(os.path.dirname(__file__))
vector_search_root = this_file_path + "/../../../"

siftsmall_file_root = vector_search_root + "external/test_data/files/siftsmall/"

siftsmall_inputs_file = siftsmall_file_root + "input_vectors.fvecs"
siftsmall_query_file = siftsmall_file_root + "queries.fvecs"
siftsmall_groundtruth_file = siftsmall_file_root + "groundtruth.ivecs"
