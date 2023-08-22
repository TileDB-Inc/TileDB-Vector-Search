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
  size_t N = 10000;

  size_t k_nn = 10;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<db_type>(ctx, fmnist_test, N);
  db.load();
//  auto db = std::move(sift_base);
//  N = db.num_cols();
//  k_nn = 2;

  auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(db, db, k_nn + 1 , 3);
  auto num_intersected = count_intersections(top_k, top_k, k_nn + 1);
  std::cout << "num_intersected: " << num_intersected << " / " << N * (k_nn + 1) << " = " << ((double)num_intersected)/((double)N * (double)k_nn) << std::endl;

  auto g = ::detail::graph::init_random_nn_graph<float>(db, k_nn);
  for (size_t i = 0; i < 10; ++i) {
    auto num_updates = nn_descent_step_all(g, db);
    std::cout << "num_updates: " << num_updates << std::endl;

    auto h = ColMajorMatrix<size_t>(k_nn + 1, N);
    for (size_t j = 0; j < N; ++j) {
      h(0, j) = j;
      get_top_k_from_heap(g.out_edges(j), std::span(&h(1,j), k_nn));
    }

    auto num_intersected = count_intersections(h, top_k, k_nn+1);
    std::cout << "num_intersected: " << num_intersected << " / " << N * (k_nn+1) << " = " << ((double)num_intersected)/((double)N * (double)(k_nn+1)) << std::endl;
  }
}