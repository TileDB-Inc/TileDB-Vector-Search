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
#include "detail/graph/nn-descent.h"
#include "query_common.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/flat/qv.h"

#include <tiledb/tiledb>

bool global_debug = false;

TEST_CASE("nn-descent: test test", "[nn-descent]") {
  REQUIRE(true);
}

TEST_CASE("nn-descent: accuracy", "[nn-descent]") {
  size_t N = 2048;
  size_t k_nn = 5;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<db_type>(ctx, db_uri, N);
  db.load();

  auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(db, db, k_nn + 1, 3);

  auto g = ::detail::graph::init_random_nn_graph<float>(db, 3*k_nn);
  for (size_t i = 0; i < 5; ++i) {
    auto num_updates = nn_descent_step_all(g, db);
    std::cout << "num_updates: " << num_updates << std::endl;

    auto h = ColMajorMatrix<size_t>(k_nn + 1, N);
    for (size_t j = 0; j < N; ++j) {
      h(0, j) = j;
      auto k = 1;
      for (auto&& [_, l] : out_edges(g, j)) {
        h(k, j) = l;
        ++k;
      }
    }
    auto num_intersected = count_intersections(h, top_k, k_nn);
    std::cout << "num_intersected: " << num_intersected << " / " << N * k_nn << " = " << ((double)num_intersected)/((double)N * (double)k_nn) << std::endl;
  }
}