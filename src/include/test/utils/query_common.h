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
#include "index/ivf_flat_index.h"
#include "index/ivf_pq_index.h"
#include "linalg.h"
#include "test/utils/array_defs.h"

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

/**
 * A data structure for holding configuration information to provide the same
 * configuration across multiple different tests.
 */
// base is 10k, learn is 25k
struct siftsmall_test_init_defaults {
  using feature_type = siftsmall_feature_type;
  using id_type = siftsmall_ids_type;
  using px_type = siftsmall_indices_type;

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

  siftsmall_test_init(
      tiledb::Context ctx,
      size_t nl,
      size_t num_subspaces = 0,
      size_t num_vectors = 0)
      : ctx_{ctx}
      , nlist(nl)
      , nprobe(std::min<size_t>(10, nlist))
      , training_set(tdbColMajorMatrix<feature_type>(
            ctx_, siftsmall_inputs_uri, num_vectors))
      , query_set(tdbColMajorMatrix<feature_type>(ctx_, siftsmall_query_uri))
      , groundtruth_set(tdbColMajorMatrix<siftsmall_groundtruth_type>(
            ctx_, siftsmall_groundtruth_uri)) {
    if constexpr (std::is_same_v<
                      IndexType,
                      ivf_flat_index<feature_type, id_type, px_type>>) {
      idx = IndexType(nlist, max_iter, tolerance);
    } else if constexpr (std::is_same_v<
                             IndexType,
                             ivf_pq_index<feature_type, id_type, px_type>>) {
      idx = IndexType(nlist, num_subspaces, max_iter, tolerance);
    } else {
      std::cout << "Unsupported index type" << std::endl;
    }

    training_set.load();
    query_set.load();
    groundtruth_set.load();

    std::vector<id_type> ids(_cpo::num_vectors(training_set));
    std::iota(begin(ids), end(ids), 0);

    std::tie(top_k_scores, top_k) = detail::flat::qv_query_heap(
        training_set, query_set, k_nn, 1, sum_of_squares_distance{});

    if constexpr (std::is_same_v<
                      IndexType,
                      ivf_flat_index<feature_type, id_type, px_type>>) {
      idx.train(training_set);
    } else if constexpr (std::is_same_v<
                             IndexType,
                             ivf_pq_index<feature_type, id_type, px_type>>) {
      idx.train_ivf(training_set);
    } else {
      std::cout << "Unsupported index type" << std::endl;
    }
    idx.add(training_set, ids);
  }

  auto get_write_read_idx() {
    std::string tmp_ivf_index_uri =
        (std::filesystem::temp_directory_path() / "tmp_ivf_index").string();
    tiledb::VFS vfs(ctx_);
    if (vfs.is_dir(tmp_ivf_index_uri)) {
      vfs.remove_dir(tmp_ivf_index_uri);
    }
    idx.write_index(ctx_, tmp_ivf_index_uri);
    auto idx0 =
        // ivf_flat_l2_index<feature_type, id_type, px_type>(ctx_,
        // tmp_ivf_index_uri);
        IndexType(ctx_, tmp_ivf_index_uri);
    return idx0;
  }

  tdbColMajorMatrix<siftsmall_feature_type> training_set;
  tdbColMajorMatrix<siftsmall_feature_type> query_set;
  tdbColMajorMatrix<siftsmall_groundtruth_type> groundtruth_set;
  ColMajorMatrix<float> top_k_scores;
  ColMajorMatrix<siftsmall_ids_type> top_k;
  // ivf_flat_l2_index<feature_type, id_type, px_type> idx;
  IndexType idx;

  auto verify(auto&& top_k_ivf) {
    // These are helpful for debugging
    // debug_matrix(top_k_ivf, "top_k_ivf");
    // debug_matrix(top_k_ivf_scores, "top_k_ivf_scores");

    size_t intersectionsm1 = count_intersections(top_k, groundtruth_set, k_nn);
    double recallm1 = intersectionsm1 / ((double)top_k.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(
          intersectionsm1 == (size_t)(num_vectors(top_k) * dimensions(top_k)));
      CHECK(recallm1 == 1.0);
    }
    CHECK(recallm1 > .99);

    // @todo There is randomness in initialization of kmeans, use a fixed seed
    size_t intersections0 = count_intersections(top_k_ivf, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(intersections0 == (size_t)(num_vectors(top_k) * dimensions(top_k)));
      CHECK(recall0 == 1.0);
    }

    size_t intersections1 =
        (long)count_intersections(top_k_ivf, groundtruth_set, k_nn);
    double recall1 = intersections1 / ((double)top_k_ivf.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(intersections1 == (size_t)(num_vectors(top_k) * dimensions(top_k)));
      CHECK(recall1 == 1.0);
    }

    if constexpr (std::is_same_v<
                      IndexType,
                      ivf_flat_index<feature_type, id_type, px_type>>) {
      CHECK(recall0 > 0.95);
      CHECK(recall1 > 0.95);

    } else if constexpr (std::is_same_v<
                             IndexType,
                             ivf_pq_index<feature_type, id_type, px_type>>) {
      CHECK(recall0 > 0.7);
      CHECK(recall1 > 0.7);

    } else {
      std::cout << "Unsupported index type" << std::endl;
    }
    // std::cout << "Recall: " << recall0 << " " << recall1 << std::endl;
  }
};

#endif  // TILEDB_QUERY_COMMON_H
