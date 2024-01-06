/**
 * @file   query_common.h
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

#ifndef TILEDB_QUERY_COMMON_H
#define TILEDB_QUERY_COMMON_H

#include <string>
#include "detail/flat/qv.h"
#include "linalg.h"

// clang-format off

  auto centroids = ColMajorMatrix<float> {
      {
          {8, 6, 7},
          {5, 3, 0},
          {9, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
          {4, 5, 6},
          {7, 8, 9},
          {3.14, 1.59, 2.65},
          {35, 89, 793},
          {2, 384, 6.26},
          {4, 33, 8},
          {32.7, 9.502, 8},
          {84, 1, 97},
          {3, 1, 4},
          {1, 5, 9},
          {9, 0, 3,},
          {5, 7, 6},
      }
  };
  auto query = ColMajorMatrix<float> {
      {
          {3, 4, 5},
          {2, 300, 8},
          {3, 1, 3.5},
          {3, 1, 3},
          {4, 5, 6},
      }
  };

  /**
   * Taken from [0:5,9:12] of sift_base
   */
  auto sift_base = ColMajorMatrix<float>
      {
          { 21.,  13.,  17.},
          { 13.,  60.,  10.},
          { 18.,  15.,   6.},
          { 11.,   4.,  47.},
          { 14.,   5.,  11.},
          {  6.,   1.,   6.},
          {  4.,   1.,   1.},
          { 14.,   9.,  20.},
          { 39.,  11.,  49.},
          { 54.,  72.,  86.},
          { 52., 114.,  36.},
          { 10.,  30.,  33.},
          {  8.,   2.,   5.},
          { 14.,   1.,   6.},
          {  5.,   9.,   2.},
          {  2.,  25.,   0.},
          { 23.,   2.,   9.},
          { 76.,  29.,  62.},
          { 65., 114.,  53.},
          { 10.,  17.,  29.},
          { 11.,   2.,  10.},
          { 23.,  12.,  19.},
          {  3.,  11.,   4.},
          {  0.,   0.,   0.},
          {  6.,   2.,   6.},
          { 10.,  33.,   9.},
          { 17.,  56.,   9.},
          {  5.,  11.,   7.},
          {  7.,   2.,   7.},
          { 21.,  35.,  30.},
          { 20.,  10.,  12.},
          { 13.,   2.,  10.},
      };

  auto sift_query = ColMajorMatrix<float>
      {
          { 0.,  7., 50.},
          {11.,  4., 43.},
          {77.,  5.,  9.},
          {24., 11.,  1.},
          { 3.,  2.,  0.}
      };

// clang-format on

#if 0
using db_type = uint8_t;

using groundtruth_type = int32_t;
using centroids_type = float;
using ids_type = uint64_t;
using indices_type = uint64_t;

static std::string ec2_root{
    "/home/lums/TileDB-Vector-Search/external/data/gp3"};
static std::string m1_root{
    "/Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3"};
static std::string db_uri{m1_root + "/1M/bigann1M_base"};
static std::string centroids_uri{m1_root + "/1M/centroids.tdb"};
static std::string parts_uri{m1_root + "/1M/parts.tdb"};
static std::string index_uri{m1_root + "/1M/index.tdb"};
static std::string sizes_uri{m1_root + "/1M/index_size.tdb"};
static std::string ids_uri{m1_root + "/1M/ids.tdb"};
static std::string query_uri{m1_root + "/1M/query_public_10k"};
static std::string groundtruth_uri{m1_root + "/1M/bigann_1M_GT_nnids"};
#else
using db_type = float;

using groundtruth_type = int32_t;
using centroids_type = float;
using ids_type = uint64_t;
using indices_type = uint64_t;

static std::string ec2_gp3_root =
    "/home/lums/TileDB-Vector-Search/external/data/gp3/";
static std::string m1_gp3_root =
    "/Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/";

static std::string ec2_root{
    "/home/lums/TileDB-Vector-Search/external/data/gp3/"};
static std::string m1_root{
    "/Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/"};
static std::string db_uri{m1_root + "sift/sift_base"};
static std::string centroids_uri{m1_root + "sift/centroids"};
static std::string parts_uri{m1_root + "sift/parts"};
static std::string index_uri{m1_root + "sift/index"};
//  static std::string sizes_uri{m1_root + "/1M/index_size.tdb"};
static std::string ids_uri{m1_root + "sift/ids"};
static std::string query_uri{m1_root + "sift/sift_query"};
static std::string groundtruth_uri{m1_root + "sift/sift_groundtruth"};

static std::string ivf_index_uri{
    m1_root + "sift/flatIVF_index_sift_base_64_64"};
static std::string ivf_index_centroids_uri{ivf_index_uri + "/centroids"};
static std::string ivf_index_indices_uri{ivf_index_uri + "/indices"};
static std::string ivf_index_ids_uri{ivf_index_uri + "/partitioned_ids"};
static std::string ivf_index_vectors_uri{
    ivf_index_uri + "/partitioned_vectors"};

static std::string bigann1M_base_uri{m1_root + "1M/bigann1M_base"};
static std::string bigann1M_query_uri{m1_root + "1M/query_public_10k"};
static std::string bigann1M_groundtruth_uri{m1_root + "1M/bigann_1M_GT_nnids"};
static std::string bigann1M_centroids_uri{m1_root + "1M/centroids.tdb"};
static std::string bigann1M_ids_uri{m1_root + "1M/ids.tdb"};
static std::string bigann1M_index_uri{m1_root + "1M/index.tdb"};
static std::string bigann1M_index_size_uri{m1_root + "1M/index_size.tdb"};
static std::string bigann1M_parts_uri{m1_root + "1M/parts.tdb"};

static std::string fmnist_train_uri{m1_root + "fmnist/fmnist_train.tdb"};
static std::string fmnist_test_uri{m1_root + "fmnist/fmnist_test.tdb"};
static std::string fmnist_groundtruth_uri{
    m1_root + "fmnist/fmnist_neighbors.tdb"};
static std::string sift_base_uri{m1_root + "sift/sift_base"};

static std::string fmnist_distances{
    m1_gp3_root + "fmnist/fmnist_distances.tdb"};
static std::string fmnist_neighbors{
    m1_gp3_root + "fmnist/fmnist_neighbors.tdb"};
static std::string fmnist_test{m1_gp3_root + "fmnist/fmnist_test.tdb"};
static std::string fmnist_train{m1_gp3_root + "fmnist/fmnist_train.tdb"};

static std::string diskann_test_256bin{
    m1_gp3_root + "diskann/siftsmall_learn_256pts.fbin"};

static std::string siftsmall_base_uri{m1_root + "siftsmall/siftsmall_base"};
static std::string siftsmall_groundtruth_uri{
    m1_root + "siftsmall/siftsmall_groundtruth"};
static std::string siftsmall_query_uri{m1_root + "siftsmall/siftsmall_query"};
static std::string siftsmall_flatIVF_index_uri{
    m1_root + "siftsmall/flatIVF_index_siftsmall_base"};
static std::string siftsmall_flatIVF_index_uri_32_64{
    m1_root + "siftsmall/flatIVF_index_siftsmall_base_32_64"};
static std::string siftsmall_flatIVF_index_uri_64_64{
    m1_root + "siftsmall/flatIVF_index_siftsmall_base_64_64"};
/*
 * siftsmall_base
 ArraySchema(
    domain=Domain(*[
      Dim(name='rows', domain=(0, 127), tile=128, dtype='int32'),
      Dim(name='cols', domain=(0, 9999), tile=10000, dtype='int32'),
    ]),
    attrs=[
      Attr(name='a', dtype='float32', var=False, nullable=False),
    ],
    cell_order='col-major',
    tile_order='col-major',
    sparse=False,
 )
 */

// /Users/lums/TileDB/TileDB-Vector-Search/external/DiskANN/rust/diskann/tests/data/

static std::string m1_diskann_root =
    "/Users/lums/TileDB/TileDB-Vector-Search/external/DiskANN/";
static std::string m1_diskann_rust_tests_root =
    m1_diskann_root + "rust/diskann/tests/";
static std::string m1_diskann_rust_test_data_root =
    m1_diskann_rust_tests_root + "data/";

static std::string diskann_test_data_file =
    m1_diskann_rust_test_data_root + "siftsmall_learn_256pts.fbin";
static std::string diskann_disk_index_path_prefix =
    m1_diskann_rust_test_data_root +
    "disk_index_siftsmall_learn_256pts_R4_L50_A1.2";
static std::string diskann_truth_disk_layout =
    m1_diskann_rust_test_data_root +
    "truth_disk_index_siftsmall_learn_256pts_R4_L50_A1.2_disk.index";
static std::string diskann_disk_index =
    diskann_disk_index_path_prefix + "_disk.index";
static std::string diskann_mem_index =
    diskann_disk_index_path_prefix + "_mem.index";
static std::string diskann_truth_index_data =
    m1_diskann_rust_test_data_root +
    "truth_index_siftsmall_learn_256pts_R4_L50_A1.2.data";
// static std::string diskann_truth_mem_layout = m1_diskann_rust_test_data_root
// + "truth_disk_index_siftsmall_learn_256pts_R4_L50_A1.2_mem.index";

auto group_uri_root = std::filesystem::path{
    "/Users/lums/TileDB/TileDB-Vector-Search/external/data/pytest-170/"};
auto group_uri_path =
    std::filesystem::path{"test_ivf_flat_ingestion_f320/array"};
auto group_uri = group_uri_root / group_uri_path;
#endif

/**
 * A data structure for holding configuration information to provide the same
 * configuration across multiple different tests.
 */
// base is 10k, learn is 25k
struct siftsmall_test_init_defaults {
  using feature_type = float;
  using id_type = uint32_t;
  using px_type = uint64_t;

  size_t k_nn = 10;
  size_t nthreads = 0;
  size_t max_iter = 10;
  float tolerance = 1e-4;
};

template <class IndexType>
struct siftsmall_test_init : public siftsmall_test_init_defaults {
  using Base = siftsmall_test_init_defaults;

  using feature_type = Base::feature_type;
  using id_type = Base::id_type;
  using px_type = Base::px_type;

  tiledb::Context ctx_;
  size_t nlist;
  size_t nprobe;

  siftsmall_test_init(tiledb::Context ctx, size_t nl)
      : ctx_{ctx}
      , nlist(nl)
      , nprobe(std::min<size_t>(10, nlist))
      , training_set(tdbColMajorMatrix<feature_type>(ctx_, siftsmall_base_uri))
      , query_set(tdbColMajorMatrix<feature_type>(ctx_, siftsmall_query_uri))
      , groundtruth_set(
            tdbColMajorMatrix<int32_t>(ctx_, siftsmall_groundtruth_uri))
      , idx(/*128,*/ nlist, max_iter, tolerance) {
    training_set.load();
    query_set.load();
    groundtruth_set.load();
    std::tie(top_k_scores, top_k) = detail::flat::qv_query_heap(
        training_set, query_set, k_nn, 1, sum_of_squares_distance{});

    idx.train(training_set);
    idx.add(training_set);
  }

  auto get_write_read_idx() {
    std::string tmp_ivf_index_uri = "/tmp/tmp_ivf_index";
    idx.write_index(ctx_, tmp_ivf_index_uri, true);
    auto idx0 =
        // ivf_flat_l2_index<feature_type, id_type, px_type>(ctx_,
        // tmp_ivf_index_uri);
        IndexType(ctx_, tmp_ivf_index_uri);
    return idx0;
  }

  tdbColMajorMatrix<float> training_set;
  tdbColMajorMatrix<float> query_set;
  tdbColMajorMatrix<int32_t> groundtruth_set;
  ColMajorMatrix<float> top_k_scores;
  ColMajorMatrix<uint64_t> top_k;
  // ivf_flat_l2_index<feature_type, id_type, px_type> idx;
  IndexType idx;

  auto verify(auto&& top_k_ivf) {
    // These are helpful for debugging
    // debug_slice(top_k_ivf, "top_k_ivf");
    // debug_slice(top_k_ivf_scores, "top_k_ivf_scores");

    auto intersectionsm1 =
        (long)count_intersections(top_k, groundtruth_set, k_nn);
    double recallm1 = intersectionsm1 / ((double)top_k.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(intersectionsm1 == num_vectors(top_k) * dimension(top_k));
      CHECK(recallm1 == 1.0);
    }
    CHECK(recallm1 > .99);

    // @todo There is randomness in initialization of kmeans, use a fixed seed
    auto intersections0 = (long)count_intersections(top_k_ivf, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(intersections0 == num_vectors(top_k) * dimension(top_k));
      CHECK(recall0 == 1.0);
    }
    CHECK(recall0 > .965);

    auto intersections1 =
        (long)count_intersections(top_k_ivf, groundtruth_set, k_nn);
    double recall1 = intersections1 / ((double)top_k_ivf.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(intersections1 == num_vectors(top_k) * dimension(top_k));
      CHECK(recall1 == 1.0);
    }
    CHECK(recall1 > 0.965);

    // std::cout << "Recall: " << recall0 << " " << recall1 << std::endl;
  }
};

#endif  // TILEDB_QUERY_COMMON_H
