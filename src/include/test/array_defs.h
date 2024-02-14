/**
 * @file   array_defs.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023 TileDB, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 *
 */

#ifndef TILEDB_ARRAY_DEFS_H
#define TILEDB_ARRAY_DEFS_H

#include <filesystem>
#include <string>

#include "config.h"

/**
 * @brief  Some default types used in unit tests.
 */

using test_feature_type = float;

using test_groundtruth_type = uint64_t;  // Should be same as ids_type
using test_centroids_type = float;
using test_ids_type = uint64_t;
using test_indices_type = uint64_t;  // Should be same as groundtruth_type

/*
 * We put the root of the test arrays in the test directory.  This needs to
 * be an absolute path, because we don't know where the test is being run.
 * We also don't want to make copies of the data under every build directory,
 * so we put it in CMAKE_SOURCE_DIR/external/test_arrays.
 */
static std::filesystem::path cmake_source_dir{CMAKE_SOURCE_DIR};
static std::filesystem::path test_data_root{
    cmake_source_dir.parent_path() / "external" / "test_data"};
static std::filesystem::path test_array_root{test_data_root / "arrays"};
static std::filesystem::path test_file_root{test_data_root / "files"};
static std::filesystem::path nano_root{test_data_root / "nano"};

/**
 * @brief  Array URIs for arrays used for unit testing of IVF indexes.
 * See http://corpus-texmex.irisa.fr for details about the datasets.
 *
 * The "sift" dataset comprises 1M float vectors of dimension 128.
 * The "siftsmall" dataset comprises 10k float vectors of dimension 128.
 * The "bigann_1M" dataset comprises 1M uint8 vectors of dimension 128.
 * The "fmnist" dataset comprises 60k uint8 vectors of dimension 784.
 *
 * These are the individual arrays.  We are not yet using a TileDB group to
 * store any of the indexes, though we are using the same names as the arrays
 * stored in the (version 0.3) group.
 *
 * The datasets are stored in their own subdirectories: "sift", "siftsmall",
 * "bigann1M", and "fmnist".  Under each subdirectory, the input vectors are
 * stored in the array "input_vectors".  If there are external ids for the
 * vectors, they are stored in the array "external_ids".  The vectors comprising
 * an IVF index include "partition_centroids", "partition_indexes",
 * "shuffled_vector_ids", and "shuffled_vectors".
 *
 * Queries used for testing are stored in the array "queries" and the groundruth
 * for each query is stored in the array "groundtruth".
 */

/*
 * Definitions for IVF index for "sift" (10k subset on
 * http://corpus-texmex.irisa.fr) Here, because of some hard-coded typing in the
 * Python implementation, groundtruth is typed as uint64_t -- however, in the
 * reference data from the website, the grountruth type is uint32_t.
 *
 * @todo Create groundtruth with uint32_t as well as ids_type and indices_type
 * as uint32_t.  Verify they can be mixed and matched.
 */
using sift_feature_type = float;
using sift_groundtruth_type = uint64_t;
using sift_centroids_type = float;
using sift_ids_type = uint64_t;
using sift_indices_type = uint64_t;
#ifdef USE_1M_UNIT_TEST_ARRAYS
constexpr size_t num_sift_vectors = 1'000'000;
static std::filesystem::path sift_root{test_array_root / "sift"};
#else
constexpr size_t num_sift_vectors = 10'000;
static std::filesystem::path sift_root{test_array_root / "siftsmall"};
#endif
constexpr size_t sift_dimension = 128;
static std::filesystem::path sift_group_uri{sift_root / "group"};
static std::filesystem::path sift_inputs_uri{sift_root / "input_vectors"};
static std::filesystem::path sift_centroids_uri{
    sift_root / "partition_centroids"};
static std::filesystem::path sift_index_uri{sift_root / "partition_indexes"};
static std::filesystem::path sift_ids_uri{sift_root / "shuffled_vector_ids"};
static std::filesystem::path sift_parts_uri{sift_root / "shuffled_vectors"};
static std::filesystem::path sift_query_uri{sift_root / "queries"};
static std::filesystem::path sift_groundtruth_uri{sift_root / "groundtruth"};

using siftsmall_feature_type = float;
using siftsmall_groundtruth_type = uint64_t;
using siftsmall_centroids_type = float;
using siftsmall_ids_type = uint64_t;
using siftsmall_indices_type = uint64_t;
constexpr size_t num_siftsmall_vectors = 10'000;
constexpr size_t siftsmall_dimension = 128;
static std::filesystem::path siftsmall_root{test_array_root / "siftsmall"};
static std::filesystem::path siftsmall_group_uri{siftsmall_root / "group"};
static std::filesystem::path siftsmall_inputs_uri{
    siftsmall_root / "input_vectors"};
static std::filesystem::path siftsmall_centroids_uri{
    siftsmall_root / "partition_centroids"};
static std::filesystem::path siftsmall_index_uri{
    siftsmall_root / "partition_indexes"};
static std::filesystem::path siftsmall_ids_uri{
    siftsmall_root / "shuffled_vector_ids"};
static std::filesystem::path siftsmall_parts_uri{
    siftsmall_root / "shuffled_vectors"};
static std::filesystem::path siftsmall_query_uri{siftsmall_root / "queries"};
static std::filesystem::path siftsmall_groundtruth_uri{
    siftsmall_root / "groundtruth"};

using siftsmall_uint8_feature_type = uint8_t;
using siftsmall_uint8_groundtruth_type = uint64_t;
using siftsmall_uint8_centroids_type = float;
using siftsmall_uint8_ids_type = uint64_t;
using siftsmall_uint8_indices_type = uint64_t;
constexpr size_t num_siftsmall_uint8_vectors = 10'000;
constexpr size_t siftsmall_uint8_dimension = 128;
static std::filesystem::path siftsmall_uint8_root{
    test_array_root / "siftsmall_uint8"};
static std::filesystem::path siftsmall_uint8_group_uri{
    siftsmall_uint8_root / "group"};
static std::filesystem::path siftsmall_uint8_inputs_uri{
    siftsmall_uint8_root / "input_vectors"};
static std::filesystem::path siftsmall_uint8_centroids_uri{
    siftsmall_uint8_root / "partition_centroids"};
static std::filesystem::path siftsmall_uint8_index_uri{
    siftsmall_uint8_root / "partition_indexes"};
static std::filesystem::path siftsmall_uint8_ids_uri{
    siftsmall_uint8_root / "shuffled_vector_ids"};
static std::filesystem::path siftsmall_uint8_parts_uri{
    siftsmall_uint8_root / "shuffled_vectors"};
static std::filesystem::path siftsmall_uint8_query_uri{
    siftsmall_uint8_root / "queries"};
static std::filesystem::path siftsmall_uint8_groundtruth_uri{
    siftsmall_uint8_root / "groundtruth"};

using bigann1M_feature_type = uint8_t;
using bigann1M_groundtruth_type = uint64_t;
using bigann1M_centroids_type = float;
using bigann1M_ids_type = uint64_t;
using bigann1M_indices_type = uint64_t;
#ifdef USE_1M_UNIT_TEST_ARRAYS
constexpr size_t num_bigann1M_vectors = 1'000'000;
static std::filesystem::path bigann1M_root{test_array_root / "bigann1M"};
#else
static std::filesystem::path bigann1M_root{test_array_root / "bigann10k"};
constexpr size_t num_bigann1M_vectors = 10'000;
#endif
constexpr size_t bigann1M_dimension = 128;
static std::filesystem::path bigann1M_group_uri{bigann1M_root / "group"};
static std::filesystem::path bigann1M_inputs_uri{
    bigann1M_root / "input_vectors"};
static std::filesystem::path bigann1M_centroids_uri{
    bigann1M_root / "partition_centroids"};
static std::filesystem::path bigann1M_index_uri{
    bigann1M_root / "partition_indexes"};
static std::filesystem::path bigann1M_ids_uri{
    bigann1M_root / "shuffled_vector_ids"};
static std::filesystem::path bigann1M_parts_uri{
    bigann1M_root / "shuffled_vectors"};
static std::filesystem::path bigann1M_query_uri{bigann1M_root / "queries"};
static std::filesystem::path bigann1M_groundtruth_uri{
    bigann1M_root / "groundtruth"};

using bigann10k_feature_type = uint8_t;
using bigann10k_groundtruth_type = uint64_t;
using bigann10k_centroids_type = float;
using bigann10k_ids_type = uint64_t;
using bigann10k_indices_type = uint64_t;
constexpr size_t num_bigann10k_vectors = 10'000;
constexpr size_t bigann10k_dimension = 128;
static std::filesystem::path bigann10k_root{test_array_root / "bigann10k"};
static std::filesystem::path bigann10k_group_uri{bigann10k_root / "group"};
static std::filesystem::path bigann10k_inputs_uri{
    bigann10k_root / "input_vectors"};
static std::filesystem::path bigann10k_centroids_uri{
    bigann10k_root / "partition_centroids"};
static std::filesystem::path bigann10k_index_uri{
    bigann10k_root / "partition_indexes"};
static std::filesystem::path bigann10k_ids_uri{
    bigann10k_root / "shuffled_vector_ids"};
static std::filesystem::path bigann10k_parts_uri{
    bigann10k_root / "shuffled_vectors"};
static std::filesystem::path bigann10k_query_uri{bigann10k_root / "queries"};
static std::filesystem::path bigann10k_groundtruth_uri{
    bigann10k_root / "groundtruth"};

using fmnistsmall_feature_type = float;
using fmnistsmall_groundtruth_type = uint64_t;
using fmnistsmall_centroids_type = float;
using fmnistsmall_ids_type = uint64_t;
using fmnistsmall_indices_type = uint64_t;
constexpr size_t num_fmnistsmall_vectors = 1'000;
constexpr size_t fmnistsmall_dimension = 784;
static std::filesystem::path fmnistsmall_root{test_array_root / "fmnistsmall"};
static std::filesystem::path fmnistsmall_group_uri{fmnistsmall_root / "group"};
static std::filesystem::path fmnistsmall_inputs_uri{
    fmnistsmall_root / "input_vectors"};
static std::filesystem::path fmnistsmall_centroids_uri{
    fmnistsmall_root / "partition_centroids"};
static std::filesystem::path fmnistsmall_index_uri{
    fmnistsmall_root / "partition_indexes"};
static std::filesystem::path fmnistsmall_ids_uri{
    fmnistsmall_root / "shuffled_vector_ids"};
static std::filesystem::path fmnistsmall_parts_uri{
    fmnistsmall_root / "shuffled_vectors"};
static std::filesystem::path fmnistsmall_query_uri{
    fmnistsmall_root / "queries"};
static std::filesystem::path fmnistsmall_groundtruth_uri{
    fmnistsmall_root / "groundtruth"};

using fmnist_feature_type = float;
using fmnist_groundtruth_type = uint64_t;
using fmnist_centroids_type = float;
using fmnist_ids_type = uint64_t;
using fmnist_indices_type = uint64_t;

#ifdef USE_1M_UNIT_TEST_ARRAYS
constexpr size_t num_fmnist_vectors = 60'000;
static std::filesystem::path fmnist_root{test_array_root / "fmnist"};
#else
constexpr size_t num_fmnist_vectors = num_fmnistsmall_vectors;
static std::filesystem::path fmnist_root{test_array_root / "fmnistsmall"};
#endif
constexpr size_t fmnist_dimension = 784;

static std::filesystem::path fmnist_group_uri{fmnist_root / "group"};
static std::filesystem::path fmnist_inputs_uri{fmnist_root / "input_vectors"};
static std::filesystem::path fmnist_centroids_uri{
    fmnist_root / "partition_centroids"};
static std::filesystem::path fmnist_index_uri{
    fmnist_root / "partition_indexes"};
static std::filesystem::path fmnist_ids_uri{
    fmnist_root / "shuffled_vector_ids"};
static std::filesystem::path fmnist_parts_uri{fmnist_root / "shuffled_vectors"};
static std::filesystem::path fmnist_query_uri{fmnist_root / "queries"};
static std::filesystem::path fmnist_groundtruth_uri{
    fmnist_root / "groundtruth"};

/**
 * @brief Some additional arrays that are not part of the IVF index, but
 * are part of the full fmnist dataset.
 */
static std::filesystem::path fmnist_train_uri{fmnist_root / "train"};
static std::filesystem::path fmnist_distances_uri{fmnist_root / "distances"};

/**
 * @brief Raw data files.  (Note:  These are files not arrays!)
 */
static std::filesystem::path siftsmall_files_root{test_file_root / "siftsmall"};
static std::filesystem::path siftsmall_inputs_file{
    siftsmall_files_root / "input_vectors.fvecs"};
static std::filesystem::path siftsmall_query_file{
    siftsmall_files_root / "queries.fvecs"};
static std::filesystem::path siftsmall_groundtruth_file{
    siftsmall_files_root / "groundtruth.ivecs"};

/**
 * @brief Data files used in unit tests in the DiskANN git repo.
 * (Note:  These are files not arrays!)
 */
static std::filesystem::path diskann_root{test_file_root / "diskann"};
static std::filesystem::path diskann_test_256bin{
    diskann_root / "siftsmall_learn_256pts.fbin"};
static std::filesystem::path diskann_test_data_file{
    diskann_root / "siftsmall_learn_256pts.fbin"};
static std::filesystem::path diskann_truth_disk_layout{
    diskann_root /
    "truth_disk_index_siftsmall_learn_256pts_R4_L50_A1.2_disk.index"};

static std::string diskann_disk_index_root_prefix{
    "disk_index_siftsmall_learn_256pts_R4_L50_A1.2"};

static std::filesystem::path diskann_disk_index{
    diskann_root / (diskann_disk_index_root_prefix + "_disk.index")};
static std::filesystem::path diskann_mem_index{
    diskann_root / (diskann_disk_index_root_prefix + "_mem.index")};

static std::filesystem::path diskann_truth_index_data =
    diskann_root / "truth_index_siftsmall_learn_256pts_R4_L50_A1.2.data";

/*
 * Definitions for the "nano" reference group for vamana index
 */
using vamana_nano_feature_type = float;
using vamana_nano_groundtruth_type = uint64_t;
using vamana_nano_ids_type = uint64_t;
using vamana_nano_indices_type = uint64_t;
constexpr size_t num_vamana_nano_vectors = 231;
constexpr size_t vamana_nano_dimension = 128;
static std::filesystem::path vamana_nano_root{nano_root / "vamana"};
static std::filesystem::path vamana_nano_group_uri{
    vamana_nano_root / "vamana_test_index"};

static std::filesystem::path vamana_nano_inputs_uri{
    vamana_nano_root / "feature_vectors"};
static std::filesystem::path vamana_nano_feature_vectors_uri{
    vamana_nano_root / "feature_vectors"};
static std::filesystem::path vamana_nano_adjacency_scores_uri{
    vamana_nano_root / "adjacency_scores"};
static std::filesystem::path vamana_nano_adjacency_ids_uri{
    vamana_nano_root / "adjacency_ids"};
static std::filesystem::path vamana_nano_adjacency_row_index_uri{
    vamana_nano_root / "adjacency_row_index"};

static std::filesystem::path vamana_nano_query_uri{
    vamana_nano_root / "queries"};
static std::filesystem::path vamana_nano_groundtruth_uri{
    vamana_nano_root / "groundtruth"};

#define TEMP_LEGACY_URIS
#ifdef TEMP_LEGACY_URIS
using db_type = siftsmall_feature_type;
using groundtruth_type = siftsmall_groundtruth_type;
using centroids_type = siftsmall_centroids_type;
using ids_type = siftsmall_ids_type;
using indices_type = siftsmall_indices_type;

static std::filesystem::path db_uri{siftsmall_root / "input_vectors"};
static std::filesystem::path centroids_uri{
    siftsmall_root / "partition_centroids"};
static std::filesystem::path index_uri{siftsmall_root / "partition_indexes"};
static std::filesystem::path ids_uri{siftsmall_root / "shuffled_vector_ids"};
static std::filesystem::path parts_uri{siftsmall_root / "shuffled_vectors"};
static std::filesystem::path query_uri{siftsmall_root / "queries"};
static std::filesystem::path groundtruth_uri{siftsmall_root / "groundtruth"};
#endif

#endif  // TILEDB_ARRAY_DEFS_H
