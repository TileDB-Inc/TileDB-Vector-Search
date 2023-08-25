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
#include "detail/graph/nn-graph.h"
#include "detail/graph/vamana.h"
#include "query_common.h"

#include <tiledb/tiledb>

bool global_debug = false;

TEST_CASE("vamana: test test", "[vamana]") {
  REQUIRE(true);
}

TEST_CASE("vamana: greedy search", "[vamana]") {
  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 7;

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

  std::vector<size_t> nbd(k_near);
  auto g = detail::graph::init_random_nn_graph<float>(nn_hypercube, k_near);
  auto query = Vector<float>{1.0, 1.0, 1.0};
  greedy_search(g, nn_hypercube, 0, query, k_near, L, nbd);

  std::cout << "Nearest neighbors:" << std::endl;
  for (auto&& n : nbd) {
    std::cout << n << " (" << nn_hypercube(0, n) << ", " << nn_hypercube(1, n)
              << ", " << nn_hypercube(2, n) << "), "
              << sum_of_squares_distance{}(nn_hypercube[n], query) << std::endl;
  }

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
}