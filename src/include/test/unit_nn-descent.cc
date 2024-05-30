/**
 * @file   unit_nn-descent.h
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

#include <catch2/catch_all.hpp>
#include "detail/flat/qv.h"
#include "detail/graph/nn-descent.h"
#include "detail/ivf/qv.h"
#include "detail/linalg/compat.h"
#include "detail/linalg/tdb_matrix.h"
#include "test/utils/array_defs.h"
#include "test/utils/query_common.h"

#include <tiledb/tiledb>

bool debug = false;

TEMPLATE_TEST_CASE(
    "accuracy", "[nn-descent]", uint32_t, int32_t, uint64_t, int64_t) {
  size_t k_nn = 10;

  using feature_type = float;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<feature_type>(ctx, fmnistsmall_inputs_uri);
  db.load();
  auto N = num_vectors(db);

  auto&& [top_k_scores, top_k] =
      detail::flat::qv_query_heap(db, db, k_nn + 1, 3);
  auto num_intersected = count_intersections(top_k, top_k, k_nn + 1);
  if (debug) {
    std::cout << "num_intersected: " << num_intersected << " / "
              << N * (k_nn + 1) << " = "
              << ((double)num_intersected) / ((double)N * (double)(k_nn + 1))
              << std::endl;
  }

  {
    scoped_timer _{"nn_descent"};
    auto g =
        ::detail::graph::init_random_nn_graph<feature_type, uint32_t>(db, k_nn);
    for (size_t i = 0; i < 4; ++i) {
      auto num_updates = nn_descent_1_step_all(g, db);
      if (debug) {
        std::cout << "num_updates: " << num_updates << std::endl;
      }

      auto h = ColMajorMatrix<size_t>(k_nn + 1, N);
      for (size_t j = 0; j < N; ++j) {
        h(0, j) = j;
        get_top_k_from_heap(g.out_edges(j), std::span(&h(1, j), k_nn));
      }
      auto num_intersected = count_intersections(h, top_k, k_nn + 1);
      if (debug) {
        std::cout << "num_intersected: " << num_intersected << " / "
                  << N * (k_nn + 1) << " = "
                  << ((double)num_intersected) /
                         ((double)N * (double)(k_nn + 1))
                  << std::endl;
      }
    }
    _.stop();
  }
}

TEMPLATE_TEST_CASE(
    "connectivity", "[nn-descent]", int32_t, uint32_t, int64_t, uint64_t) {
  size_t k_nn = 10;

  using feature_type = float;
  using id_type = TestType;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<feature_type>(ctx, fmnistsmall_inputs_uri);
  db.load();
  auto N = num_vectors(db);

  auto g = detail::graph::nn_descent_1<feature_type, id_type>(db, k_nn);

  bfs(g, TestType{0});
}

TEST_CASE("nn_descent_1", "[nn-descent]") {
  using feature_type = float;
  using id_type = uint32_t;

  size_t nthreads = 1;
  size_t k_nn = 10;
  size_t num_queries = 10;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<feature_type>(ctx, fmnistsmall_inputs_uri);
  db.load();
  auto N = num_vectors(db);

  auto g = detail::graph::nn_descent_1<feature_type, id_type>(db, k_nn);
  auto query = ColMajorMatrix<float>(db.num_rows(), num_queries);
  for (size_t i = 0; i < db.num_rows(); ++i) {
    query(i, 0) = db(i, 0);
  }

  log_timer flat_timer{"flat_query", true};
  auto&& [top_s, top_k] =
      detail::flat::qv_query_heap(db, query, k_nn + 1, nthreads);
  flat_timer.stop();

  std::vector<size_t> tv(k_nn + 1);
  for (size_t i = 0; i < k_nn + 1; ++i) {
    tv[i] = top_k(i, 0);
  }

  std::vector<float> sv(k_nn + 1);
  for (size_t i = 0; i < k_nn + 1; ++i) {
    sv[i] = top_s(i, 0);
  }

  log_timer query_timer{"nn_descent_1_query", true};
  auto&& [s, t] = nn_descent_1_query(g, db, query, k_nn, k_nn + 5, 3);
  query_timer.stop();

  std::vector<size_t> tw(k_nn + 1);
  for (size_t i = 0; i < k_nn + 1; ++i) {
    tw[i] = t(i, 0);
  }
  std::vector<size_t> sw(k_nn + 1);
  for (size_t i = 0; i < k_nn + 1; ++i) {
    sw[i] = s(i, 0);
  }
  std::sort(begin(tv), end(tv));
  std::sort(begin(tw), end(tw));

  auto tt = ColMajorMatrix<size_t>(k_nn + 1, 1);
  for (size_t j = 0; j < 1; ++j) {
    tt(0, j) = j;
    for (size_t i = 0; i < k_nn; ++i) {
      tt(i + 1, j) = t(i, j);
    }
  }

  auto num_intersected = count_intersections(tt, top_k, k_nn + 1);
  if (debug) {
    std::cout << "num_intersected: " << num_intersected << " / "
              << num_queries * (k_nn + 1) << " = "
              << ((double)num_intersected) /
                     ((double)num_queries * (double)(k_nn + 1))
              << std::endl;
  }
}

TEST_CASE("nn_descent_1 vs ivf", "[nn-descent]") {
  using feature_type = float;
  using id_type = uint64_t;

  size_t nthreads = 1;
  size_t k_nn = 10;
  size_t num_queries = 50;

  tiledb::Context ctx;

  auto db = tdbColMajorMatrix<feature_type>(ctx, sift_inputs_uri);
  db.load();
  auto N = num_vectors(db);

  auto centroids = tdbColMajorMatrix<feature_type>(ctx, sift_centroids_uri);
  centroids.load();
  auto query =
      tdbColMajorMatrix<feature_type>(ctx, sift_query_uri, num_queries);
  query.load();
  auto index = read_vector<test_indices_type>(ctx, sift_index_uri);
  auto groundtruth =
      tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
  groundtruth.load();

  auto parts = tdbColMajorMatrix<feature_type>(ctx, sift_parts_uri);
  parts.load();
  auto ids = read_vector<uint64_t>(ctx, sift_ids_uri);

  size_t nprobe = 20;

  // nlog_timer flat_timer{"flat_query", true};
  // auto&& [top_s, top_k] = detail::flat::qv_query_heap(db, query, k_nn + 1,
  // nthreads); flat_timer.stop();

  // @todo Use updated partitioned_matrix interface rather than compat

  log_timer ivf_timer{"ivf_query", true};
  auto mat = ColMajorPartitionedMatrixWrapper<feature_type, id_type, id_type>(
      parts, ids, index);

  auto&& [active_partitions, active_queries] =
      detail::ivf::partition_ivf_flat_index<id_type>(
          centroids, query, nprobe, nthreads);

  //  auto&& [D00, I00] = detail::ivf::query_infinite_ram(
  //      parts, centroids, query, index, ids, nprobe, k_nn + 1, nthreads);
  auto&& [D00, I00] = detail::ivf::query_infinite_ram(
      mat, active_partitions, query, active_queries, k_nn + 1, nthreads);
  ivf_timer.stop();

  log_timer graph_timer{"nn_descent_1", true};
  auto g = detail::graph::nn_descent_1<feature_type, id_type>(db, k_nn);
  graph_timer.stop();

  log_timer query_timer{"nn_descent_1_query", true};
  auto&& [s, t] = nn_descent_1_query(g, db, query, k_nn, k_nn + 5, 3);
  query_timer.stop();

  auto tt = ColMajorMatrix<size_t>(k_nn + 1, 1);
  for (size_t j = 0; j < 1; ++j) {
    tt(0, j) = j;
    for (size_t i = 0; i < k_nn; ++i) {
      tt(i + 1, j) = t(i, j);
    }
  }

  auto num_intersected = count_intersections(t, groundtruth, k_nn + 1);
  auto qv_intersected = count_intersections(I00, groundtruth, k_nn + 1);
  if (debug) {
    std::cout << "qv_intersected: " << qv_intersected << " / "
              << num_queries * (k_nn + 1) << " = "
              << ((double)qv_intersected) /
                     ((double)num_queries * (double)(k_nn + 1))
              << std::endl;
    std::cout << "num_intersected: " << num_intersected << " / "
              << num_queries * (k_nn + 1) << " = "
              << ((double)num_intersected) /
                     ((double)num_queries * (double)(k_nn + 1))
              << std::endl;
  }
}
