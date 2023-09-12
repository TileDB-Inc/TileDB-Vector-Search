/**
 * @file   unit_vamana.h
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
#include "detail/graph/nn-graph.h"
#include "detail/graph/vamana.h"
#include "detail/graph/adj_list.h"
#include "query_common.h"
#include "utils/logging.h"
#include "graphs/tiny.h"
#include "cpos.h"

#include <tiledb/tiledb>

bool global_debug = false;

TEST_CASE("vamana: test test", "[vamana]") {
  REQUIRE(true);
}

TEST_CASE("vamana: tiny greedy search", "[vamana]") {
  auto A = index_adj_list{tiny_index_adj_list};
  size_t source = 0;
  std::vector<size_t> query {0, 1};
  size_t k = 3;
  size_t L = 3;
  std::vector<size_t> nbd(k);

  auto V = greedy_search(A, tiny_vectors, source, query, k, L, nbd);
}

auto build_hypercube(size_t k_near, size_t k_far) {
  size_t N = 8 * (k_near + k_far + 1);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist_near(-0.1, 0.1);
  std::uniform_real_distribution<float> dist_far(0.2, 0.3);
  std::uniform_int_distribution<int> heads(0, 1);

  ColMajorMatrix<float> nn_hypercube(3, N + 1);
  size_t n{0};
  nn_hypercube(0, n) = 0;
  nn_hypercube(1, n) = 0;
  nn_hypercube(2, n) = 0;
  ++n;

  for (auto i : {-1, 1}) {
    for (auto j : {-1, 1}) {
      for (auto k : {-1, 1}) {
        nn_hypercube(0, n) = i;
        nn_hypercube(1, n) = j;
        nn_hypercube(2, n) = k;
        ++n;
      }
    }
  }

  for (size_t m = 0; m < k_near; ++m) {
    for (auto i : {-1, 1}) {
      for (auto j : {-1, 1}) {
        for (auto k : {-1, 1}) {
          nn_hypercube(0, n) = i + dist_near(gen);
          nn_hypercube(1, n) = j + dist_near(gen);
          nn_hypercube(2, n) = k + dist_near(gen);
          ++n;
        }
      }
    }
  }

  for (size_t m = 0; m < k_far; ++m) {
    for (auto i : {-1, 1}) {
      for (auto j : {-1, 1}) {
        for (auto k : {-1, 1}) {
          nn_hypercube(0, n) = i + (heads(gen) ? 1 : -1) * dist_far(gen);
          nn_hypercube(1, n) = j + (heads(gen) ? 1 : -1) * dist_far(gen);
          nn_hypercube(2, n) = k + (heads(gen) ? 1 : -1) * dist_far(gen);
          ++n;
        }
      }
    }
  }

  if (global_debug) {
    std::cout << "Hypercube stats:" << std::endl;
    std::cout << "  num_rows: " << nn_hypercube.num_rows() << " ";
    std::cout << "  num_cols: " << nn_hypercube.num_cols() << std::endl;

    std::cout << "Hypercube (transpose):" << std::endl;
    for (size_t j = 0; j < nn_hypercube.num_cols(); ++j) {
      for (size_t i = 0; i < nn_hypercube.num_rows(); ++i) {
        std::cout << nn_hypercube(i, j) << ", ";
      }
      std::cout << std::endl;
    }
  }
  return nn_hypercube;
}


TEST_CASE("vamana: greedy search", "[vamana]") {
  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 5;

  auto nn_hypercube = build_hypercube(k_near, k_far);

  std::vector<size_t> nbd(k_near);
  auto g = detail::graph::init_random_nn_graph<float>(nn_hypercube, k_near);

  for (float kk : {-1, 1}) {
    auto query = Vector<float>{kk*1.05f, kk*0.95f, 1.09};
    auto V = greedy_search(g, nn_hypercube, 2, query, k_near, L, nbd);

    std::cout << "Nearest neighbors:" << std::endl;
    for (auto&& n : nbd) {
      std::cout << n << " (" << nn_hypercube(0, n) << ", " << nn_hypercube(1, n)
                << ", " << nn_hypercube(2, n) << "), "
                << sum_of_squares_distance{}(nn_hypercube[n], query)
                << std::endl;
    }
    std::cout << "-----\ntop_k\n";

    auto query_mat = ColMajorMatrix<float>(3, 1);
    for (size_t i = 0; i < 3; ++i) {
      query_mat(i, 0) = query[i];
    }
    auto&& [top_scores, top_k] =
        detail::flat::qv_query_heap(nn_hypercube, query_mat, k_near, 1);
    for (size_t i = 0; i < k_near; ++i) {
      std::cout << top_k(i, 0) << " (" << nn_hypercube(0, top_k(i, 0)) << ", "
                << nn_hypercube(1, top_k(i, 0)) << ", "
                << nn_hypercube(2, top_k(i, 0)) << "), " << top_scores(i, 0)
                << std::endl;
    }
    for (auto&& v : V) {
      std::cout << v << ", ";
    }
    std::cout << std::endl;
  }
}


TEST_CASE("vamana: greedy search with nn descent", "[vamana]") {
  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 7;

  auto nn_hypercube = build_hypercube(k_near, k_far);

  std::vector<size_t> nbd(k_near);
  auto g = detail::graph::init_random_nn_graph<float>(nn_hypercube, k_near);

  for (size_t kk : {-1, 1}) {
    auto query = Vector<float>{kk*1.05f, kk*0.95f, 1.09f};

    auto V = greedy_search(g, nn_hypercube, 0UL, query, k_near, L, nbd);
    std::cout << "size(V): " << size(V) << std::endl;

    for (size_t i = 0; i < 4; ++i) {
      auto num_updates = nn_descent_1_step_all(g, nn_hypercube);
      std::cout << "num_updates: " << num_updates << std::endl;
      if (num_updates == 0) {
        break;
      }
      V = greedy_search(g, nn_hypercube, 0UL, query, k_near, L, nbd);
      std::cout << "size(V): " << size(V) << std::endl;
    }

    for (auto&& v : V) {
      std::cout << v << ", ";
    }
    std::cout << std::endl;


    std::cout << "Nearest neighbors:" << std::endl;
    for (auto&& n : nbd) {
      std::cout << n << " (" << nn_hypercube(0, n) << ", " << nn_hypercube(1, n)
                << ", " << nn_hypercube(2, n) << "), "
                << sum_of_squares_distance{}(nn_hypercube[n], query) << std::endl;
    }
    std::cout << "-----\n";


  }
}

TEST_CASE("vamana: diskann fbin", "[vamana]") {

  size_t k_nn = 5;
  size_t L = 3;

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

  binary_file.read((char*)x.data(), npoints*ndim);
  binary_file.close();

  auto q0 = ColMajorMatrix<float> (ndim, 1);
  std::copy(x.data(), x.data() + ndim, q0.data());

  auto g = detail::graph::init_random_nn_graph<float>(x, k_nn);
  auto nbd = std::vector<size_t>(k_nn);
  auto V = greedy_search(g, x, 0UL, x[0], k_nn, L, nbd);
  REQUIRE(size(V) == 1);
}

TEST_CASE("vamana: fmnist", "[vamana]") {
  size_t nthreads = 1;
  size_t k_nn = 50;
  size_t L = 7;
  size_t num_queries = 10;
  size_t N = 5000;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<db_type>(ctx, fmnist_test, N);
  db.load();
  auto g = detail::graph::init_random_nn_graph<float>(db, L);

  auto query = db[599];

  auto query_mat = ColMajorMatrix<float>(size(query), 1);
  for (size_t i = 0; i < size(query); ++i) {
    query_mat(i, 0) = query[i];
  }

  auto&& [top_scores, top_k] =
      detail::flat::qv_query_heap(db, query_mat, k_nn, 1);
  std::sort(begin(top_k[0]), end(top_k[0]));
  std::cout << "Neighbors: ";
  for (size_t i = 0; i < k_nn; ++i) {
    std::cout << top_k(i, 0) << " ";
  }
  std::cout << "\nDistances: ";
  for (size_t i = 0; i < k_nn; ++i) {
    std::cout << sum_of_squares_distance{}(db[top_k(i,0)], query) << " ";
  }
  std::cout << "\n-----\n";

  auto nbd = ColMajorMatrix<size_t>(k_nn, 1);

  auto V = greedy_search(g, db, 0UL, query, k_nn, L, nbd[0]);
  std::cout << "size(V): " << size(V) << std::endl;
  for (size_t i = 0; i < 7; ++i) {
    auto num_updates = nn_descent_1_step_all(g, db);
    std::cout << "num_updates: " << num_updates << std::endl;
    V = greedy_search(g, db, 0UL, query, k_nn, L, nbd[0]);
    std::cout << "size(V): " << size(V) << std::endl;

    auto num_intersected = count_intersections(nbd, top_k, k_nn);
    std::cout << "num_intersected: " << num_intersected << " / " << k_nn
              << " = "
              << ((double) num_intersected) / ((double) query_mat.num_cols() * k_nn)
              << std::endl;

    std::cout << "Greedy nearest neighbors: ";
    for (auto&& n : nbd[0]) {
      std::cout << n << " ";
    }
    std::cout << "\nGreedy distances: ";
    for (auto&& n : nbd[0]) {
      std::cout << sum_of_squares_distance{}(db[n], query) << " ";
    }
    std::cout << "\n-----\n";

    if (num_updates == 0) {
      break;
    }
  }

  auto&& [s, t] = nn_descent_1_query(g, db, query_mat, k_nn, k_nn + 5, 3);

  auto num_intersected = count_intersections(t, top_k, k_nn);
  std::cout << "num_intersected: " << num_intersected << " / " << k_nn
            << " = "
            << ((double) num_intersected) / ((double) query_mat.num_cols() * k_nn)
            << std::endl;

  std::cout << "NN-descent nearest neighbors: ";
  for (size_t i = 0; i < k_nn; ++i) {
    std::cout << t(i, 0) << " ";
  }

  std::cout << "\nNN-descent returned distances: ";
  for (size_t i = 0; i < k_nn; ++i) {
    std::cout << s(i,0) << " ";
  }
  std::cout << "\nNN-descent computed distances: ";
  for (size_t i = 0; i < k_nn; ++i) {
    std::cout << sum_of_squares_distance{}(db[t(i,0)], query) << " ";
  }
  std::cout << std::endl;

}

TEST_CASE("vamana: robust prune", "[vamana]") {
  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 7;
  size_t R = 7;

  float alpha = 1.0;

  auto nn_hypercube = build_hypercube(k_near, k_far);
  auto start = medioid(nn_hypercube);
  for (auto&& s : nn_hypercube[start]) {
    std::cout << s << ", ";
  }
  std::cout << std::endl;

  std::vector<size_t> nbd(k_near);
  auto g = detail::graph::init_random_nn_graph<float>(nn_hypercube, R);
  auto query = Vector<float>{1.05, 0.95, 1.09};

  SECTION("One node") {
    auto p = 8UL;

    for (auto&& [s, t] : g.out_edges(8UL)) {
      std::cout << " ( " << t << ", " << s << " ) ";
    }
    std::cout << std::endl;

    auto V = greedy_search(g, nn_hypercube, start, nn_hypercube[p], k_near, L, nbd);
    robust_prune(g, nn_hypercube, p, V, alpha, R);

    for (auto&& [s, t] : g.out_edges(8UL)) {
      std::cout << " ( " << t << ", " << s << " ) ";
    }
    std::cout << std::endl;
  }

  SECTION("One pass") {

    for (size_t p = 0; p < nn_hypercube.num_cols(); ++p) {
      for (auto&& [s, t] : g.out_edges(p)) {
        std::cout << " ( " << t << ", " << s << " ) ";
      }
      std::cout << std::endl;

      auto V = greedy_search(g, nn_hypercube, start, nn_hypercube[p], k_near, L, nbd);
      robust_prune(g, nn_hypercube, p, V, alpha, R);

      for (auto&& [s, t] : g.out_edges(p)) {
        std::cout << " ( " << t << ", " << s << " ) ";
      }
      std::cout << std::endl;
    }

    auto V = greedy_search(g, nn_hypercube, start, query, k_near, L, nbd);
    std::cout << "V.size: " << size(V) << std::endl;
    for (auto&& v : V) {
      std::cout << v << ", ";
    }
    std::cout << std::endl;
    for (auto&& n : nbd) {
      std::cout << n << " (" << nn_hypercube(0, n) << ", " << nn_hypercube(1, n)
                << ", " << nn_hypercube(2, n) << "), "
                << sum_of_squares_distance{}(nn_hypercube[n], query) << std::endl;
    }
  }
}


TEST_CASE("vamana: robust prune fmnist", "[vamana]") {
  size_t nthreads = 1;
  size_t k_nn = 50;
  size_t L = 7;
  size_t R = 7;
  size_t num_queries = 10;
  size_t N = 500;
  float alpha = 1.0;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<db_type>(ctx, fmnist_test, N);
  db.load();
  auto g = detail::graph::init_random_nn_graph<float>(db, L);

  auto query = db[599];

  auto query_mat = ColMajorMatrix<float>(size(query), 1);
  for (size_t i = 0; i < size(query); ++i) {
    query_mat(i, 0) = query[i];
  }
  auto qv_timer = log_timer{"qv", true};
  auto&& [top_scores, top_k] =
      detail::flat::qv_query_heap(db, query_mat, k_nn, 1);
  std::sort(begin(top_k[0]), end(top_k[0]));
  qv_timer.stop();

  auto start = medioid(db);

  for (float alpha : {1.0, 1.25}) {
    for (size_t p = 0; p < db.num_cols(); ++p) {
      auto nbd = std::vector<size_t>(k_nn);
      auto V = greedy_search(g, db, start, db[p], k_nn, L, nbd);
      robust_prune(g, db, p, V, alpha, R);
    }
  }

  auto nbd = ColMajorMatrix<size_t>(k_nn, 1);

  auto greedy_timer = log_timer{"greedy", true};
  auto V = greedy_search(g, db, start, query, k_nn, L, nbd[0]);
  greedy_timer.stop();

  std::cout << "V.size: " << size(V) << std::endl;

  // for (auto&& v : V) {
  //   std::cout << v << ", ";
  // }

  std::cout << std::endl;
  for (auto&& n : nbd[0]) {
    std::cout << n << " (" << db(0, n) << ", " << db(1, n) << ", " << db(2, n)
              << "), " << sum_of_squares_distance{}(db[n], query) << std::endl;
  }
  auto num_intersected = count_intersections(nbd, top_k, k_nn);
  std::cout << "num_intersected: " << num_intersected << " / " << k_nn << " = "
            << ((double)num_intersected) / ((double)query_mat.num_cols() * k_nn)
            << std::endl;
}


