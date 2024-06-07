/**
 * @file   unit_linalg.cc
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
 */

#include <catch2/catch_all.hpp>
#include "detail/graph/nn-graph.h"
#include "test/utils/query_common.h"

TEST_CASE("init_random_graph", "[nn-graph]") {
  using feature_type = float;
  using id_type = uint32_t;

  auto k_nn = 5UL;
  auto g = ::detail::graph::init_random_nn_graph<feature_type, id_type>(
      sift_base, k_nn);

  CHECK(g.num_vertices() == sift_base.num_cols());
  size_t total_degree = 0;
  for (size_t i = 0; i < g.num_vertices(); ++i) {
    CHECK(detail::graph::out_degree(g, i) != 0);
    CHECK(detail::graph::out_degree(g, i) == k_nn);
    total_degree += detail::graph::out_degree(g, i);
  }
  CHECK(total_degree == g.num_vertices() * k_nn);
}

TEST_CASE("reverse random graph", "[nn-graph]") {
  using feature_type = float;
  using id_type = uint32_t;

  auto k_nn = 5UL;
  auto g = ::detail::graph::init_random_nn_graph<feature_type, id_type>(
      sift_base, k_nn);
  CHECK(g.num_vertices() == sift_base.num_cols());
  size_t total_out_degree = 0;
  for (size_t i = 0; i < g.num_vertices(); ++i) {
    CHECK(detail::graph::out_degree(g, i) != 0);
    CHECK(detail::graph::out_degree(g, i) == k_nn);
    total_out_degree += detail::graph::out_degree(g, i);
  }
  CHECK(total_out_degree == g.num_vertices() * k_nn);
  g.build_in_edges();
  size_t total_in_degree = 0;
  for (size_t i = 0; i < g.num_vertices(); ++i) {
    total_in_degree += detail::graph::in_degree(g, i);
  }
  CHECK(total_in_degree == g.num_vertices() * k_nn);
}
