/**
 * @file unit_best_first.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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
#include <filesystem>
#include "detail/graph/adj_list.h"
#include "detail/graph/best_first.h"
#include "test/utils/gen_graphs.h"
#include "test/utils/tiny_graphs.h"

TEST_CASE("best first", "[best_first]") {
  size_t num_vertices = 5;
  detail::graph::adj_list<size_t, size_t> graph(num_vertices);
  for (size_t src = 0; src < num_vertices; ++src) {
    for (size_t dst = 0; dst < num_vertices; ++dst) {
      auto score = static_cast<size_t>(
          std::pow(static_cast<int>(src) - static_cast<int>(dst), 2));
      graph.add_edge(src, dst, score);
    }
  }
  CHECK(detail::graph::num_vertices(graph) == num_vertices);
  CHECK(graph.num_vertices() == num_vertices);
  std::vector<std::tuple<size_t, size_t>> expected = {
      {0, 0}, {1, 1}, {4, 2}, {9, 3}, {16, 4}};
  CHECK(
      std::equal(expected.begin(), expected.end(), graph.out_edges(0).begin()));

  auto feature_vectors = ColMajorMatrixWithIds<float, size_t>{
      {{1.f, 1.f, 1.f},
       {2.f, 2.f, 2.f},
       {3.f, 3.f, 3.f},
       {4.f, 4.f, 4.f},
       {5.f, 5.f, 5.f}},
      {1, 2, 3, 4, 5}};

  size_t k_nn = 1;
  size_t source = 0;
  {
    auto visited =
        best_first_O0(graph, feature_vectors, source, feature_vectors[0]);
    CHECK(visited == std::unordered_set<size_t>{0, 1, 2, 3, 4});
  }

  {
    auto visited =
        best_first_O1(graph, feature_vectors, source, feature_vectors[0]);
    CHECK(visited == std::unordered_set<size_t>{0, 1, 2, 3, 4});
  }

  // NOTE: Here is where you would expect to also test best_first_O2 and
  // best_first_O3, but they do  not compile. We'll leave them for now, but
  // eventually we should either remove them or get them building.

  // Test best_first_O4 when used normally.
  for (size_t l_max = 1; l_max < 6; ++l_max) {
    auto&& [top_k_scores, top_k, visited] = best_first_O4(
        graph, feature_vectors, source, feature_vectors[0], k_nn, l_max);

    std::unordered_set<size_t> expected_visited;
    for (size_t i = 0; i < l_max; ++i) {
      expected_visited.insert(i);
    }
    CHECK(visited == expected_visited);
    CHECK(top_k_scores.size() == k_nn);
    CHECK(top_k.size() == k_nn);
    CHECK(top_k_scores[0] == 0);
    CHECK(top_k[0] == 0);
  }

  for (size_t l_max = 1; l_max < 6; ++l_max) {
    auto&& [top_k_scores, top_k, visited] = best_first_O5(
        graph, feature_vectors, source, feature_vectors[0], k_nn, l_max);

    std::unordered_set<size_t> expected_visited;
    for (size_t i = 0; i < l_max; ++i) {
      expected_visited.insert(i);
    }
    CHECK(visited == expected_visited);
    CHECK(top_k_scores.size() == k_nn);
    CHECK(top_k.size() == k_nn);
    CHECK(top_k_scores[0] == 0);
    CHECK(top_k[0] == 0);
  }

  // Test best_first_O4 when with skip_top_k=true.
  for (size_t l_max = 1; l_max < 6; ++l_max) {
    auto&& [top_k_scores, top_k, visited] = best_first_O4(
        graph, feature_vectors, source, feature_vectors[0], k_nn, l_max, true);

    std::unordered_set<size_t> expected_visited;
    for (size_t i = 0; i < l_max; ++i) {
      expected_visited.insert(i);
    }
    CHECK(visited == expected_visited);
    // In this case we do not compute top_k and so these are empty.
    CHECK(top_k_scores.empty());
    CHECK(top_k.empty());
  }
}
