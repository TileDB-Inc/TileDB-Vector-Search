/**
* @file   graph.h
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


#ifndef TILEDB_GRAPH_H
#define TILEDB_GRAPH_H

#include "scoring.h"
#include "utils/fixed_min_heap.h"
#include <vector>

namespace detail::graph {

template <class T, class I = size_t>
class nn_graph {
  size_t num_vertices_{0};
  size_t k_nn_{0};

  std::vector<fixed_min_pair_heap<T, I>> out_edges_;

 public:
  nn_graph(size_t num_vertices, size_t k_nn)
      : num_vertices_{num_vertices}
      , k_nn_{k_nn}
      , out_edges_(num_vertices, fixed_min_pair_heap<T, I>(k_nn)) {
  }

  void add_edge(I src, I dst, T score) {
    out_edges_[src].insert(score, dst);
  }

  void add_edge(const fixed_min_pair_heap<T, I> neighbors, I dst, T score) {
    neighbors.insert(score, dst);
  }

  auto& out_edges(I src) const {
    return out_edges_[src];
  }

  auto num_vertices() const {
    return num_vertices_;
  }
};

template <class T, class I>
auto num_vertices(const nn_graph<T, I>& g) {
  return g.num_vertices();
}

template <class T, class I>
auto out_edges(const nn_graph<T, I>& g, size_t i) {
  return g.out_edges(i);
}

template <class T, class I>
auto make_random_nn_graph(auto&& db, size_t k_nn) {
  auto num_vertices = db.num_cols();
  nn_graph<T, I> g(num_vertices, k_nn);
  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t i = 0; i < num_vertices; ++i) {
    auto thisvec = db[i];

    std::uniform_int_distribution<> dis(0, num_vertices - 1);
    for (size_t j = 0; j < k_nn - 1; ++j) {
      auto nbr = dis(gen);
        nbr = dis(gen);
      auto score = L2(thisvec, db[nbr]);
      g.add_edge(i, nbr, score);
    }
  }
  return g;
}
}  // namespace detail::graph


#endif  // TILEDB_GRAPH_H