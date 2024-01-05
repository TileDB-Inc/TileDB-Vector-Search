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

using test_groundtruth_type = int32_t;
using test_centroids_type = float;
using test_ids_type = uint64_t;
using test_indices_type = uint64_t;

/*
 * We put the root of the test arrays in the test directory.  This needs to
 * be an absolute path, because we don't know where the test is being run.
 * We also don't want to make copies of the data under every build directory,
 * so we put it in CMAKE_SOURCE_DIR/external/test_arrays.
 */
static std::filesystem::path cmake_source_dir { CMAKE_SOURCE_DIR };
static std::filesystem::path test_data_root { cmake_source_dir.parent_path() / "external" / "test_data" };
static std::filesystem::path test_array_root { test_data_root / "arrays" };
static std::filesystem::path test_file_root { test_data_root / "files" };

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
static std::filesystem::path sift_root{test_array_root / "sift"};
static std::filesystem::path sift_inputs_uri{sift_root / "input_vectors"};
static std::filesystem::path sift_centroids_uri{
    sift_root / "partition_centroids"};
static std::filesystem::path sift_index_uri{sift_root / "partition_indexes"};
static std::filesystem::path sift_ids_uri{sift_root / "shuffled_vector_ids"};
static std::filesystem::path sift_parts_uri{sift_root / "shuffled_vectors"};
static std::filesystem::path sift_query_uri{sift_root / "queries"};
static std::filesystem::path sift_groundtruth_uri{sift_root / "groundtruth"};

static std::filesystem::path siftsmall_root{test_array_root / "siftsmall"};
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

static std::filesystem::path siftsmall_uint8_root{test_array_root / "siftsmall_uint8"};
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
static std::filesystem::path siftsmall_uint8_query_uri{siftsmall_uint8_root / "queries"};
static std::filesystem::path siftsmall_uint8_groundtruth_uri{
    siftsmall_uint8_root / "groundtruth"};

static std::filesystem::path bigann1M_root{test_array_root / "bigann1M"};
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

static std::filesystem::path bigann10k_root{test_array_root / "bigann10k"};
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

static std::filesystem::path fmnist_root{test_array_root / "fmnist"};
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
static std::filesystem::path diskann_root{test_array_root / "diskann"};
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

static std::filesystem::path pytest_170_group_root{test_array_root / "pytest-170"};
static std::filesystem::path pytest_170_group_uri{
    pytest_170_group_root / "test_ivf_flat_ingestion_f320/array"};

#endif  // TILEDB_ARRAY_DEFS_H