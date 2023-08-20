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
  std::vector<std::set<I>> in_edges_;
  std::vector<fixed_min_pair_heap<T, I>> update_edges_;

public:
  nn_graph(size_t num_vertices, size_t k_nn)
          : num_vertices_{num_vertices}
          , k_nn_{k_nn}
          , out_edges_(num_vertices, fixed_min_pair_heap<T, I>(k_nn))
          , in_edges_(num_vertices)
          , update_edges_(num_vertices, fixed_min_pair_heap<T, I>(k_nn))
  { }


  /**
   * Attempt to add edge to out_edge neighbor list.  Note that we can't add a corresponding edge to
   * the in_edges_ list because we don't know which edge may have been displaced if the insert was
   * successful.
   * @param src
   * @param dst
   * @param score
   */
  auto add_edge(I src, I dst, T score) {
    return out_edges_[src]. template insert<typename std::remove_cvref_t<decltype(out_edges_[src])>::unique_id>(score, dst);
  }

  /**
   * Attempt to add edge to update_edge neighbor list.  Note that we can't add a corresponding edge to
   * the in_edges_ list because we don't know which edge may have been displaced if the insert was
   * successful.
   * @param src
   * @param dst
   * @param score
   */
  auto add_update_edge(I src, I dst, T score) {
    return update_edges_[src]. template insert<typename std::remove_cvref_t<decltype(out_edges_[src])>::unique_id>(score, dst);
  }

  auto copy_to_update_edges() {
    for (size_t i = 0; i < num_vertices_; ++i) {
      for (auto&& [s, e] : out_edges_[i]) {
        update_edges_[i].insert(s, e);
      }
    }
  }

  auto swap_update_edges(I i) {
    out_edges_[i].swap(update_edges_[i]);
  }

  auto swap_all_update_edges() {
    for (I i = 0; i < num_vertices_; ++i) {
      swap_update_edges(i);
    }
  }

  auto build_in_edges() {
    for (size_t i = 0; i < num_vertices_; ++i) {
      in_edges_[i].clear();
    }
    for (size_t i = 0; i < num_vertices_; ++i) {
      for (auto && [s, e] : out_edges_[i]) {
        in_edges_[e].insert(i);
      }
    }
  }

  auto& out_edges(I src) const {
    return out_edges_[src];
  }

  auto& in_edges(I dst) const {
    return in_edges_[dst];
  }

  auto num_vertices() const {
    return num_vertices_;
  }

  auto out_degree(I src) const {
    return out_edges_[src].size();
  }

  auto in_degree(I dst) const {
    return in_edges_[dst].size();
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
auto in_edges(const nn_graph<T, I>& g, size_t i) {
  return g.in_edges(i);
}

template <class T, class I>
auto out_degree(const nn_graph<T, I>& g, size_t i) {
  return g.out_degree(i);
}

template <class T, class I>
auto in_degree(const nn_graph<T, I>& g, size_t i) {
  return g.in_degree(i);
}



/**
 * @brief Initialize a random graph with k_nn edges per vertex.  Each vertex is connected to another
 * vertex chosen at random with a uniform probablity.  The graph is a directed graph, so the edges are
 * the out edges of each vertex.
 * @tparam T
 * @tparam I
 * @param db
 * @param k_nn
 * @return
 */
template <class T, class I = size_t, class Distance = sum_of_squares_distance>
auto init_random_nn_graph(auto&& db, size_t k_nn, Distance distance = Distance()) {
  auto num_vertices = db.num_cols();
  nn_graph<T, I> g(num_vertices, k_nn);
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<size_t> dis(0, num_vertices - 1);

  for (size_t i = 0; i < num_vertices; ++i) {
    for (size_t j = 0; j < k_nn; ++j) {
      auto nbr = dis(gen);
      while (nbr == i) {
        nbr = dis(gen);
      }
      auto score = distance(db[i], db[nbr]);
      while (g.add_edge(i, nbr, score) == false) {
        nbr = dis(gen);
        while (nbr == i) {
          nbr = dis(gen);
        }
        score = distance(db[i], db[nbr]);
      }
    }
  }
  return g;
}


}  // namespace detail::graph


#endif  // TILEDB_GRAPH_H