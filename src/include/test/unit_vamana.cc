/**
 * @file   unit_vamana.cc
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

#include "cpos.h"
#include "detail/flat/qv.h"
#include "detail/graph/adj_list.h"
#include "detail/graph/nn-descent.h"
#include "detail/graph/nn-graph.h"
#include "detail/graph/vamana.h"
#include "gen_graphs.h"
#include "graphs/tiny.h"
#include "query_common.h"
#include "utils/logging.h"

#include <tiledb/tiledb>

bool global_debug = false;

TEST_CASE("vamana: test test", "[vamana]") {
  REQUIRE(true);
}

#if 0
TEST_CASE("vamana: tiny greedy search", "[vamana]") {
  auto A = index_adj_list{tiny_index_adj_list};
  size_t source = 0;
  std::vector<size_t> query {0, 1};
  size_t k = 3;
  size_t L = 3;
  std::vector<size_t> top_k(k);

  auto V = greedy_search(A, tiny_vectors, source, query, k, L);
}
#endif

TEST_CASE("vamana: greedy grid search", "[vamana]") {
  const bool debug = false;

  size_t M = 5;
  size_t N = 7;

  auto one_two = GENERATE(1, 2);
  auto&& [vecs, edges] = ([one_two, M, N]() {
    if (one_two == 1) {
      return gen_uni_grid(M, N);
    } else {
      return gen_bi_grid(M, N);
    }
  })();

  auto expected_size = ((M - 1) * N + M * (N - 1)) * one_two;
  CHECK(num_vectors(vecs) == M * N);
  CHECK(edges.size() == expected_size);

  detail::graph::adj_list<float, size_t> A(35);
  for (auto&& [src, dst] : edges) {
    CHECK(src < A.num_vertices());
    CHECK(dst < A.num_vertices());
    A.add_edge(src, dst, sum_of_squares_distance{}(vecs[src], vecs[dst]));
  }

  // (2, 3): 17 -> {10, 16, 17, 18, 24}
  // (3, 4): 25 -> {18, 24, 25, 26, 32}
  // (3, 6): 27 -> {20, 26, 27, 34}
  // (4, 5): 33 -> {26, 32, 33, 34}
  // (4, 6): 33 -> {27, 33, 34}

  using expt_type = std::tuple<
      std::vector<size_t>,
      size_t,
      std::vector<size_t>,
      size_t,
      size_t,
      std::vector<size_t>>;
  std::vector<expt_type> expts{
      {{0, 0}, 0, {2, 3}, 17, 7, {10, 16, 17, 18, 24}},
      {{0, 0}, 0, {3, 4}, 25, 9, {18, 24, 25, 26, 32}},
      {{0, 0}, 0, {3, 6}, 27, 11, {20, 26, 27, 34}},  // 33
      {{0, 0}, 0, {4, 5}, 33, 11, {26, 32, 33, 34}},  // 25
      {{0, 0}, 0, {4, 6}, 34, 11, {27, 33, 34}},      // 26 32
  };

  size_t count = 0;
  for (
      auto&& [source_vec, source_coord, query_vec, query_coord, path_length, expected] :
      expts) {
    size_t k = 5;
    size_t L = 5;  // --> L must be >= k
    size_t expected_length =
        query_vec[0] - source_vec[0] + query_vec[1] - source_vec[1] + 2;

    auto&& [top_k_scores, top_k, V] =
        greedy_search(A, vecs, source_coord, query_vec, size(expected), L);
    CHECK(size(top_k) == size(expected));

    std::sort(begin(top_k), begin(top_k) + size(expected));
    CHECK(std::equal(begin(expected), end(expected), begin(top_k)));

    if (debug) {
      std::cout << ":::: " << count << " :::: \n";
      for (auto&& n : top_k) {
        std::cout << n << " ";
      }
      std::cout << std::endl;
      for (auto&& e : expected) {
        std::cout << e << " ";
      }
      std::cout << std::endl;
      for (auto&& v : V) {
        std::cout << "(" << vecs[v][0] << ", " << vecs[v][1] << ") ";
      }
      std::cout << std::endl;
    }
  }
}

TEST_CASE("vamana: greedy search hypercube", "[vamana]") {
  const bool debug = false;

  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 5;

  auto nn_hypercube = build_hypercube(k_near, k_far);

  auto g = detail::graph::init_random_nn_graph<float>(nn_hypercube, k_near);

  for (float kk : {-1, 1}) {
    auto query = Vector<float>{kk * 1.05f, kk * 0.95f, 1.09};
    auto [top_k_scores, top_k, V] =
        greedy_search(g, nn_hypercube, 2, query, k_near, L);

    if (debug) {
      std::cout << "Nearest neighbors:" << std::endl;
      for (auto&& n : top_k) {
        std::cout << n << " (" << nn_hypercube(0, n) << ", "
                  << nn_hypercube(1, n) << ", " << nn_hypercube(2, n) << "), "
                  << sum_of_squares_distance{}(nn_hypercube[n], query)
                  << std::endl;
      }
      std::cout << "-----\ntop_k\n";
    }

    auto query_mat = ColMajorMatrix<float>(3, 1);
    for (size_t i = 0; i < 3; ++i) {
      query_mat(i, 0) = query[i];
    }

    {
      auto&& [top_k_scores, top_k] =
          detail::flat::qv_query_heap(nn_hypercube, query_mat, k_near, 1);

      if (debug) {
        for (size_t i = 0; i < k_near; ++i) {
          std::cout << top_k(i, 0) << " (" << nn_hypercube(0, top_k(i, 0))
                    << ", " << nn_hypercube(1, top_k(i, 0)) << ", "
                    << nn_hypercube(2, top_k(i, 0)) << "), "
                    << top_k_scores(i, 0) << std::endl;
        }
        for (auto&& v : V) {
          std::cout << v << ", ";
        }
        std::cout << std::endl;
      }
    }
  }
}

TEST_CASE("vamana: greedy search with nn descent", "[vamana]") {
  const bool debug = false;

  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 7;

  auto nn_hypercube = build_hypercube(k_near, k_far);

  std::vector<std::tuple<float, size_t>> top_k(k_near);
  auto g = detail::graph::init_random_nn_graph<float>(nn_hypercube, k_near);

  auto valid = validate_graph(g, nn_hypercube);
  CHECK(valid.size() == 0);

  for (size_t kk : {-1, 1}) {
    auto query = Vector<float>{kk * 1.05f, kk * 0.95f, 1.09f};

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, nn_hypercube, 0UL, query, k_near, L);

    if (debug) {
      std::cout << "size(V): " << size(V) << std::endl;
    }

    for (size_t i = 0; i < 4; ++i) {
      auto num_updates = nn_descent_1_step_all(g, nn_hypercube);
      if (debug) {
        std::cout << "num_updates: " << num_updates << std::endl;
      }
      if (num_updates == 0) {
        break;
      }
      auto&& [top_k_scores, top_k, V] =
          greedy_search(g, nn_hypercube, 0UL, query, k_near, L);

      if (debug) {
        std::cout << "size(V): " << size(V) << std::endl;
      }
    }

    if (debug) {
      for (auto&& v : V) {
        std::cout << v << ", ";
      }
      std::cout << std::endl;

      std::cout << "Nearest neighbors:" << std::endl;
      for (auto&& n : top_k) {
        std::cout << n << " (" << nn_hypercube(0, n) << ", "
                  << nn_hypercube(1, n) << ", " << nn_hypercube(2, n) << "), "
                  << sum_of_squares_distance{}(nn_hypercube[n], query)
                  << std::endl;
      }
      std::cout << "-----\n";
    }
  }
}

TEST_CASE("vamana: diskann fbin", "[vamana]") {
  size_t k_nn = 5;
  size_t L = 5;

  // should be dim = 128, num = 256
  // npoints, ndims
  uint32_t npoints{0};
  uint32_t ndim{0};

  std::ifstream binary_file(diskann_test_256bin, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error("Could not open file " + diskann_test_256bin);
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);
  REQUIRE(npoints == 256);
  REQUIRE(ndim == 128);

  auto x = ColMajorMatrix<float>(ndim, npoints);

  binary_file.read((char*)x.data(), npoints * ndim);
  binary_file.close();

  SECTION("vary starts") {
    auto start = GENERATE(0, 17, 127, 128, 129, 254, 255);

    auto g = detail::graph::init_random_nn_graph<float>(x, k_nn);

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, x, start, x[start], k_nn, L);
    std::sort(begin(top_k), end(top_k));

    // CHECK(top_k[0] == start);
    CHECK(std::find(begin(top_k), end(top_k), start) != end(top_k));
    CHECK(std::find(begin(V), end(V), start) != end(V));
  }
}

TEST_CASE("vamana: fmnist", "[vamana]") {
  const bool debug = false;

  size_t nthreads = 1;
  size_t L = 7;
  size_t k_nn = L;
  size_t num_queries = 10;
  size_t N = 5000;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<db_type>(ctx, fmnist_test, N);
  db.load();
  auto g = detail::graph::init_random_nn_graph<float>(db, L);

  auto valid = validate_graph(g, db);
  CHECK(valid.size() == 0);

  auto query = db[599];

  auto query_mat = ColMajorMatrix<float>(size(query), 1);
  for (size_t i = 0; i < size(query); ++i) {
    query_mat(i, 0) = query[i];
  }

  auto&& [top_scores, qv_top_k] =
      detail::flat::qv_query_heap(db, query_mat, k_nn, 1);
  std::sort(begin(qv_top_k[0]), end(qv_top_k[0]));

  if (debug) {
    std::cout << "Neighbors: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << qv_top_k(i, 0) << " ";
    }
    std::cout << "\nDistances: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << sum_of_squares_distance{}(db[qv_top_k(i, 0)], query) << " ";
    }
    std::cout << "\n-----\n";
  }
  auto&& [top_k_scores, top_k, V] = greedy_search(g, db, 0UL, query, k_nn, L);

  if (debug) {
    std::cout << "size(V): " << size(V) << std::endl;
  }
  for (size_t i = 0; i < 7; ++i) {
    auto num_updates = nn_descent_1_step_all(g, db);

    if (debug) {
      std::cout << "num_updates: " << num_updates << std::endl;
    }

    auto&& [top_k_scores, top_k, V] = greedy_search(g, db, 0UL, query, k_nn, L);

    auto valid = validate_graph(g, db);
    CHECK(valid.size() == 0);

    if (debug) {
      std::cout << "size(V): " << size(V) << std::endl;
    }

    auto top_n = ColMajorMatrix<size_t>(k_nn, 1);
    for (size_t i = 0; i < k_nn; ++i) {
      top_n(i, 0) = top_k[i];
    }

    auto num_intersected = count_intersections(top_n, qv_top_k, k_nn);

    if (debug) {
      std::cout << "num_intersected: " << num_intersected << " / " << k_nn
                << " = "
                << ((double)num_intersected) /
                       ((double)query_mat.num_cols() * k_nn)
                << std::endl;

      std::cout << "Greedy nearest neighbors: ";
      for (auto&& n : top_k) {
        std::cout << n << " ";
      }
      std::cout << "\nGreedy distances: ";
      for (auto&& n : top_k) {
        std::cout << sum_of_squares_distance{}(db[n], query) << " ";
      }
      std::cout << "\n-----\n";
    }

    if (num_updates == 0) {
      break;
    }
  }

  auto&& [s, t] = nn_descent_1_query(g, db, query_mat, k_nn, k_nn + 5, 3);

  auto num_intersected = count_intersections(t, qv_top_k, k_nn);

  if (debug) {
    std::cout << "num_intersected: " << num_intersected << " / " << k_nn
              << " = "
              << ((double)num_intersected) /
                     ((double)query_mat.num_cols() * k_nn)
              << std::endl;

    std::cout << "NN-descent nearest neighbors: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << t(i, 0) << " ";
    }

    std::cout << "\nNN-descent returned distances: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << s(i, 0) << " ";
    }
    std::cout << "\nNN-descent computed distances: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << sum_of_squares_distance{}(db[t(i, 0)], query) << " ";
    }
    std::cout << std::endl;
  }
}

TEST_CASE("vamana: robust prune hypercube", "[vamana]") {
  bool debug = false;

  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 7;
  size_t R = 7;

  float alpha = 1.0;

  auto nn_hypercube = build_hypercube(k_near, k_far);
  auto start = detail::graph::medioid(nn_hypercube);

  if (debug) {
    for (auto&& s : nn_hypercube[start]) {
      std::cout << s << ", ";
    }
    std::cout << std::endl;
  }

  std::vector<std::tuple<float, size_t>> top_k(k_near);
  auto g = detail::graph::init_random_nn_graph<float>(nn_hypercube, R);

  auto valid = validate_graph(g, nn_hypercube);
  CHECK(valid.size() == 0);

  auto query = Vector<float>{1.05, 0.95, 1.09};

  SECTION("One node") {
    auto p = 8UL;

    if (debug) {
      for (auto&& [s, t] : g.out_edges(8UL)) {
        std::cout << " ( " << t << ", " << s << " ) ";
      }
      std::cout << std::endl;
    }

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, nn_hypercube, start, nn_hypercube[p], k_near, L);
    robust_prune(g, nn_hypercube, p, V, alpha, R);

    if (debug) {
      for (auto&& [s, t] : g.out_edges(8UL)) {
        std::cout << " ( " << t << ", " << s << " ) ";
      }
      std::cout << std::endl;
    }
  }

  SECTION("One pass") {
    for (size_t p = 0; p < nn_hypercube.num_cols(); ++p) {
      if (debug) {
        for (auto&& [s, t] : g.out_edges(p)) {
          std::cout << " ( " << t << ", " << s << " ) ";
        }
        std::cout << std::endl;
      }

      auto&& [top_k_scores, top_k, V] =
          greedy_search(g, nn_hypercube, start, nn_hypercube[p], k_near, L);

      auto valid = validate_graph(g, nn_hypercube);
      CHECK(valid.size() == 0);

      robust_prune(g, nn_hypercube, p, V, alpha, R);

      auto valid2 = validate_graph(g, nn_hypercube);
      CHECK(valid2.size() == 0);

      if (debug) {
        for (auto&& [s, t] : g.out_edges(p)) {
          std::cout << " ( " << t << ", " << s << " ) ";
        }
        std::cout << std::endl;
      }
    }

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, nn_hypercube, start, query, k_near, L);

    auto valid = validate_graph(g, nn_hypercube);
    CHECK(valid.size() == 0);

    if (debug) {
      std::cout << "V.size: " << size(V) << std::endl;
      for (auto&& v : V) {
        std::cout << v << ", ";
      }
      std::cout << std::endl;
      for (auto&& n : top_k) {
        std::cout << n << " (" << nn_hypercube(0, n) << ", "
                  << nn_hypercube(1, n) << ", " << nn_hypercube(2, n) << "), "
                  << sum_of_squares_distance{}(nn_hypercube[n], query)
                  << std::endl;
      }
    }
  }
}

TEST_CASE("vamana: robust prune fmnist", "[vamana]") {
  bool debug = false;

  size_t nthreads = 1;
  size_t k_nn = 5;
  size_t L = 7;
  size_t R = 7;
  size_t num_queries = 10;
  size_t N = 500;
  float alpha = 1.0;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<db_type>(ctx, fmnist_test, N);
  db.load();
  auto g = detail::graph::init_random_nn_graph<float>(db, L);

  auto valid = validate_graph(g, db);
  REQUIRE(valid.size() == 0);

  auto x = std::accumulate(begin(db[0]), end(db[0]), 0.0F);

  auto query = db[N / 2 + 3];
  std::vector<float> query_vec(begin(query), end(query));

  auto query_mat = ColMajorMatrix<float>(size(query), 1);
  for (size_t i = 0; i < size(query); ++i) {
    query_mat(i, 0) = query[i];
  }

  auto valid6 = validate_graph(g, db);
  REQUIRE(valid6.size() == 0);

  auto qv_timer = log_timer{"qv", true};
  auto&& [top_scores, qv_top_k] =
      detail::flat::qv_query_heap(db, query_mat, k_nn, 1);
  std::sort(begin(qv_top_k[0]), end(qv_top_k[0]));
  qv_timer.stop();

  auto valid2 = validate_graph(g, db);
  REQUIRE(valid2.size() == 0);

  auto start = detail::graph::medioid(db);

  for (float alpha : {1.0, 1.25}) {
    if (debug) {
      std::cout << ":::: alpha: " << alpha << std::endl;
    }
    for (size_t p = 0; p < db.num_cols(); ++p) {
      auto valid3 = validate_graph(g, db);
      REQUIRE(valid3.size() == 0);

      auto&& [top_k_scores, top_k, V] =
          greedy_search(g, db, start, db[p], k_nn, L);

      auto valid4 = validate_graph(g, db);
      CHECK(valid4.size() == 0);

      robust_prune(g, db, p, V, alpha, R);

      auto valid5 = validate_graph(g, db);
      CHECK(valid5.size() == 0);
    }
  }

  auto greedy_timer = log_timer{"greedy", true};
  auto&& [top_k_scores, top_k, V] = greedy_search(g, db, start, query, k_nn, L);
  greedy_timer.stop();

  if (debug) {
    std::cout << "V.size: " << size(V) << std::endl;

    // for (auto&& v : V) {
    //   std::cout << v << ", ";
    // }
  }

  auto top_n = ColMajorMatrix<size_t>(k_nn, 1);
  for (size_t i = 0; i < k_nn; ++i) {
    top_n(i, 0) = top_k[i];
  }

  auto num_intersected = count_intersections(top_n, qv_top_k, k_nn);

  if (debug) {
    std::cout << "num_intersected: " << num_intersected << " / " << k_nn
              << " = "
              << ((double)num_intersected) /
                     ((double)query_mat.num_cols() * k_nn)
              << std::endl;

    std::cout << "Greedy nearest neighbors: ";
    for (auto&& n : top_k) {
      std::cout << n << " ";
    }
    std::cout << "\nGreedy distances: ";
    for (auto&& n : top_k) {
      std::cout << sum_of_squares_distance{}(db[n], query) << " ";
    }
    std::cout << "\n-----\n";
  }

#if 0
  std::cout << std::endl;
  for (auto&& n : top_k[0]) {
    std::cout << n << " (" << db(0, n) << ", " << db(1, n) << ", " << db(2, n)
              << "), " << sum_of_squares_distance{}(db[n], query) << std::endl;
  }
  auto num_intersected = count_intersections(top_k, top_k, k_nn);
  std::cout << "num_intersected: " << num_intersected << " / " << k_nn << " = "
            << ((double)num_intersected) / ((double)query_mat.num_cols() * k_nn)
            << std::endl;
#endif
}

TEST_CASE("vamana: vamana_index vector diskann_test_256bin", "[vamana]") {
  bool debug = false;

  // should be dim = 128, num = 256
  // npoints, ndims
  uint32_t npoints{0};
  uint32_t ndim{0};

  std::ifstream binary_file(diskann_test_256bin, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error("Could not open file " + diskann_test_256bin);
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);
  REQUIRE(npoints == 256);
  REQUIRE(ndim == 128);

  auto x = ColMajorMatrix<float>(ndim, npoints);

  binary_file.read((char*)x.data(), npoints * ndim);
  binary_file.close();

  size_t L = 100;
  size_t R = 100;
  size_t B = 2;
  auto index =
      detail::graph::vamana_index<decltype(x)>(num_vectors(x), L, R, B);

  auto x0 = std::vector<float>(ndim);
  std::copy(x.data(), x.data() + ndim, begin(x0));

  index.train(x);

  // x, 5, 7, 7, 1.0, 1, 1);
  // 0
  // 14
  auto&& [s0, v0] = index.query(x0, 5);
  CHECK(v0[0] == 0);
}

TEST_CASE("vamana: vamana by hand random index", "[vamana]") {
  const bool debug = false;

  size_t num_nodes = 20;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t L_build_ = 2;
  size_t R_max_degree_ = 2;

  auto training_set_ = random_geometric_2D(num_nodes);
  dump_coordinates("coords.txt", training_set_);

  auto g = ::detail::graph::init_random_adj_list<float, size_t>(
      training_set_, R_max_degree_);

  auto dimension_ = ::dimension(training_set_);
  auto num_vectors_ = ::num_vectors(training_set_);
  auto graph_ = ::detail::graph::init_random_nn_graph<float>(
      training_set_, R_max_degree_);

  auto medioid_ = detail::graph::medioid(training_set_);

  if (debug) {
    std::cout << "medioid: " << medioid_ << std::endl;
  }

  size_t counter{0};
  for (float alpha : {alpha_0, alpha_1}) {
    for (size_t p = 0; p < num_vectors_; ++p) {
      if (debug) {
        dump_edgelist("edges_" + std::to_string(counter++) + ".txt", graph_);
      }

      auto&& [top_k_scores, top_k, visited] = greedy_search(
          graph_, training_set_, medioid_, training_set_[p], 1, L_build_);

      if (debug) {
        std::cout << ":::: Post search prune" << std::endl;
      }
      robust_prune(graph_, training_set_, p, visited, alpha, R_max_degree_);

      for (auto&& [i, j] : graph_.out_edges(p)) {
        if (debug) {
          std::cout << ":::: Checking neighbor " << j << std::endl;
        }

        // @todo Do this without copying -- prune should take vector of tuples
        // and p (it copies anyway)
        auto tmp = std::vector<size_t>(graph_.out_degree(j) + 1);
        tmp.push_back(p);
        for (auto&& [_, k] : graph_.out_edges(j)) {
          if (k != p) {
            tmp.push_back(k);
          }
        }

        if (size(tmp) > R_max_degree_) {
          if (debug) {
            std::cout << ":::: Pruning neighbor " << j << std::endl;
          }
          robust_prune(graph_, training_set_, j, tmp, alpha, R_max_degree_);
        } else {
          graph_.add_edge(
              j,
              p,
              sum_of_squares_distance()(training_set_[p], training_set_[j]));
        }
      }
    }
  }
}

/**
 * This test recapitulates the 200 node 2D graph in the DiskANN paper
 */
TEST_CASE("vamana: vamana_index geometric 2D graph", "[vamana]") {
  const bool debug = false;

  size_t num_nodes = 200;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t L_build = 15;
  size_t R_max_degree = 15;

  size_t k_nn = 5;

  auto training_set = random_geometric_2D(num_nodes);

  auto idx = detail::graph::vamana_index<decltype(training_set)>(
      num_vectors(training_set), L_build, R_max_degree, 0);
  idx.train(training_set);

  auto query = training_set[17];
  auto&& [scores, top_k] = idx.query(query, k_nn);
  CHECK(top_k[0] == 17);

  auto query_mat = ColMajorMatrix<float>(dimension(training_set), 7);
  size_t counter{0};
  for (size_t i : {17, 19, 23, 37, 49, 50, 195}) {
    std::copy(
        training_set[i].data(),
        training_set[i].data() + dimension(training_set),
        query_mat[counter++].data());
  }

  auto&& [qv_scores, qv_top_k] =
      detail::flat::qv_query_heap(training_set, query_mat, k_nn, 4);
  auto&& [mat_scores, mat_top_k] = idx.query(query_mat, k_nn);
  size_t total_intersected = count_intersections(mat_top_k, qv_top_k, k_nn);

  if (debug) {
    std::cout << total_intersected << " / " << k_nn * num_vectors(query_mat)
              << " = "
              << ((double)total_intersected) /
                     ((double)k_nn * num_vectors(query_mat))
              << std::endl;
  }
}

TEST_CASE("vamana: vamana_index siftsmall", "[vamana]") {
  bool debug = false;

  size_t num_nodes = 10000;
  size_t num_queries = 200;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t L_build = 15;
  size_t R_max_degree = 12;

  size_t k_nn = 10;

  tiledb::Context ctx;
  auto training_set =
      tdbColMajorMatrix<float>(ctx, siftsmall_base_uri, num_nodes);
  training_set.load();
  auto queries =
      tdbColMajorMatrix<float>(ctx, siftsmall_query_uri, num_queries);
  queries.load();

  auto idx = detail::graph::vamana_index<decltype(training_set)>(
      num_vectors(training_set), L_build, R_max_degree, 0);
  idx.train(training_set);

  auto&& [qv_scores, qv_top_k] =
      detail::flat::qv_query_heap(training_set, queries, k_nn, 4);
  auto&& [mat_scores, mat_top_k] = idx.query(queries, k_nn);
  size_t total_intersected = count_intersections(mat_top_k, qv_top_k, k_nn);

  auto recall =
      ((double)total_intersected) / ((double)k_nn * num_vectors(queries));
  CHECK(recall > 0.90);  // @todo -- had been 0.95?

  if (debug) {
    std::cout << total_intersected << " / " << k_nn * num_vectors(queries)
              << " = "
              << ((double)total_intersected) /
                     ((double)k_nn * num_vectors(queries))
              << std::endl;
  }
}
