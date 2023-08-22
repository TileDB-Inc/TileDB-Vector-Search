/**
 * @file   nn-descent.h
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

#ifndef TILEDB_NN_DESCENT_H
#define TILEDB_NN_DESCENT_H

#include "scoring.h"
#include "detail/graph/nn-graph.h"

namespace detail::graph {
/*
 Basic nn-search algorithm

1. Choose a starting node in the graph (potentially randomly) as a candidate
node
2. Look at all nodes connected by an edge to the best untried candidate node in
the graph
3. Add all these nodes to our potential candidate pool
4. Sort the candidate pool by closeness / similarity to the query point
5. Truncate the pool to 𝑘 best (as in closest to the query) candidates
6. Return to step 2, unless we have already tried all the candidates in the pool

  */

/*
    NN-descent algorithm
    1. Start with a random graph (connect each node to 𝑘 random nodes)
    2. For each node:
    3. Measure the distance from the node to the neighbors of its neighbors
    4. If any are closer, then update the graph accordingly, and keep only the 𝑘
   closest
    5. If any updates were made to the graph then go back to step 2, otherwise
   stop
 */



template <class I = size_t, class Distance = sum_of_squares_distance>
auto nn_descent_step(auto&& g, auto&& db, I i, I j, Distance distance = Distance()) {
  size_t num_updates {0};
  for (auto&& [_, k] : out_edges(g, j)) {  // _ is dist(j, k)
    if (i == k) {
      continue;
    }
    auto ik_score = distance(db[i], db[k]);
    if (g.add_edge(i, k, ik_score)) {
      ++num_updates;
    }
  }
  for (auto&& k : in_edges(g, j)) {  // _ is dist(j, k)
    if (i == k) {
      continue;
    }
    auto ik_score = distance(db[i], db[k]);
    if (g.add_edge(i, k, ik_score)) {
      ++num_updates;
    }
  }

  return num_updates;
}

template <class Distance = sum_of_squares_distance>
auto nn_descent_step_all(auto&& g, auto&& db, Distance distance = Distance()) {
  size_t num_updates{0};
  size_t nvertices{num_vertices(g)};
  // g.copy_to_update_edges();

  g.build_in_edges();

  for (size_t i = 0; i < nvertices; ++i) {
    for (auto&& [_, j] : out_edges(g, i)) {
      num_updates += nn_descent_step(g, db, i, j);
    }

    for (auto&& j : in_edges(g, i)) {
      num_updates += nn_descent_step(g, db, i, j);
    }
  }
  // g.swap_all_update_edges();

  return num_updates;
}


template <class T, class I = size_t, class Distance = sum_of_squares_distance>
auto nn_descent(auto&& db, size_t k_nn, Distance distance = Distance()) {
  auto g =
      init_random_nn_graph<T, I, Distance>(db, k_nn, distance);

  size_t num_updates{0};
  do {
    num_updates = nn_descent_step_all(g, db);
  } while(num_updates > (k_nn * num_vertices(g)) / 100);

  return g;
}



}  // namespace detail::graph

#endif  // TILEDB_NN_DESCENT_H