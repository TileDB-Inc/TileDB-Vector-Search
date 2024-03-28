/**
 * @file   adj_list.h
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

#ifndef TDB_ADJ_LIST_H
#define TDB_ADJ_LIST_H

#include <initializer_list>
#include <list>
#include <vector>
#include "scoring.h"

namespace detail::graph {
template <class I>
class index_adj_list : public std::vector<std::list<I>> {
  using Base = std::vector<std::list<I>>;

 public:
  using id_type = I;
  using score_type = float;

  index_adj_list(size_t num_vertices)
      : Base(num_vertices) {
  }

#if 1
  index_adj_list(const std::vector<std::tuple<I, I>>& edge_list) {
    for (auto& [src, dst] : edge_list) {
      Base::operator[](src).push_back(dst);
    }
  }
#endif

  template <class AdjList>
  index_adj_list(AdjList&& l)
      : Base(size(l)) {
    for (size_t i = 0; i < size(l); ++i) {
      for (auto& dst : l[i]) {
        add_edge(i, dst);
      }
    }
  }

  auto add_edge(I src, I dst) {
    Base::operator[](src).push_back(dst);
  }

  auto& out_edges(I i) {
    return Base::operator[](i);
  }

  auto out_degree(I i) const {
    return Base::operator[](i).size();
  }

  auto out_degree(I i) {
    return Base::operator[](i).size();
  }

  auto& num_vertices() {
    return Base::size();
  }
};

// Deduction guide
template <class Inner>
index_adj_list(const std::vector<Inner>& l)
    -> index_adj_list<typename Inner::value_type>;

template <class I = size_t>
auto num_vertices(index_adj_list<I>& g) {
  return g.num_vertices();
}

template <class I = size_t>
auto& out_edges(index_adj_list<I>& g, I i) {
  return g.out_edges(i);
}

template <class I = size_t>
auto& out_degree(index_adj_list<I>& g, I i) {
  return g.out_degree(i);
}

/**
 * Naive adjacency list graph
 *
 * @tparam I
 *
 * @todo Optimize for performance
 */
template <class SC, std::integral ID>
class adj_list : public std::vector<std::list<std::tuple<SC, ID>>> {
  using Base = std::vector<std::list<std::tuple<SC, ID>>>;
  size_t num_edges_{0};

 public:
  using id_type = ID;
  using score_type = SC;

  explicit adj_list(size_t num_vertices = 0)
      : Base(num_vertices) {
  }

#if 0
  template <class EdgeList>
  index_adj_list(EdgeList&& edge_list) {
    for (auto& [src, dst] : edge_list) {
      Base::operator[](src).push_back(dst);
    }
  }
#endif

  template <class AdjList>
    requires(!std::integral<std::remove_cvref_t<AdjList>>)
  adj_list(AdjList&& l)
      : Base(l.size()) {
    for (size_t i = 0; i < l.size(); ++i) {
      for (auto&& [val, dst] : l[i]) {
        add_edge(i, dst, val);
      }
    }
  }

  auto add_edge(id_type src, id_type dst, const score_type& val) {
    Base::operator[](src).emplace_back(val, dst);
    ++num_edges_;
  }

  constexpr auto& out_edges(id_type i) {
    return Base::operator[](i);
  }

  constexpr auto& out_edges(id_type i) const {
    return Base::operator[](i);
  }

  constexpr auto out_degree(id_type i) {
    return Base::operator[](i).size();
  }

  constexpr auto out_degree(id_type i) const {
    return Base::operator[](i).size();
  }

  constexpr auto num_vertices() const {
    return Base::size();
  }

  constexpr auto num_edges() const {
    return num_edges_;
  }
};

template <class T, std::integral I>
auto num_vertices(const adj_list<T, I>& g) {
  return g.num_vertices();
}

template <class T, std::integral I>
auto& out_edges(const adj_list<T, I>& g, I i) {
  return g.out_edges(i);
}

template <class T, std::integral ID>
auto& out_degree(adj_list<T, ID>& g, ID i) {
  return g.out_degree(i);
}

template <class T, std::integral ID, class Distance = sum_of_squares_distance>
auto init_random_adj_list(auto&& db, size_t R, Distance distance = Distance()) {
  auto num_vertices = num_vectors(db);
  adj_list<T, ID> g(num_vertices);
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<size_t> dis(0, num_vertices - 1);

  for (size_t i = 0; i < num_vertices; ++i) {
    for (size_t j = 0; j < R; ++j) {
      auto nbr = dis(gen);
      while (nbr == i) {
        nbr = dis(gen);
      }
      auto score = distance(db[i], db[nbr]);
      g.add_edge(i, nbr, score);
    }
  }
  return g;
}

}  // namespace detail::graph
#endif  // TDB_ADJ_LIST_H
