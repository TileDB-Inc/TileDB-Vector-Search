/**
 * @file   vamana.h
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

#ifndef TDB_VAMANA_H
#define TDB_VAMANA_H

#include <cstddef>

#include <functional>
#include <queue>
#include <set>

#include "scoring.h"
#include "utils/fixed_min_heap.h"
#include "utils/print_types.h"

namespace {
  enum class SearchPath { path_and_search, path_only };
}

/**
 * @brief
 * @tparam T
 * @tparam I
 * @param graph
 * @param source start node index
 * @param query query node index
 * @param k result size
 * @param L search list size, L >= k
 * @return result set ell containing k-approximate nearest neighbors to query
 * and set vee containing all the visited nodes
 *
 * Per the paper, the algorithm is as follows:
 * 1. Initialize the result list with the source node and visited list with
 * empty
 * 2. While the result list \ visited list is not empty
 *    a. Find p* in the result list \ visited list with the smallest distance to
 * the query b. update the result list with the out neighbors of p* c. Add p* to
 * the visited list d. If size of the result list > L, trim the result list to
 * keep L closest points to query
 * 3. Copy the result list to the output
 */
template <SearchPath SP, class I = size_t, class Distance = sum_of_squares_distance>
auto greedy_search(
    auto&& graph,
    auto&& db,
    I source,
    auto&& query,
    size_t k,
    size_t L,
    auto&& nbd,
    Distance&& distance = Distance{}) {
  using value_type = typename std::decay_t<decltype(graph)>::value_type;

  std::set<I> visited_vertices;
  auto visited = [&visited_vertices](auto&& v) {
    return visited_vertices.find(v) != visited_vertices.end();
  };

  auto result = k_min_heap<value_type, I>{k};
  auto q1 = k_min_heap<value_type, I>{L};

  // L <- {s} and V <- empty`
  result.insert(distance(db[source], query), source);

  // q1 = L \ V = {s}
  q1.insert(distance(db[source], query), source);

  // while L\V is not empty
  while (!q1.empty()) {
    // p* <- argmin_{p \in L\V} distance(p, q)

    // @todo: There must be a better way to do this
    // Change to min_heap
    std::make_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) > std::get<0>(b);
    });

    // Get and pop the min element
    std::pop_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) > std::get<0>(b);
    });

    auto [s_star, p_star] = q1.back();
    q1.pop_back();

    // Change back to max heap
    std::make_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });

    if (visited(p_star)) {
      continue;
    }

    // V <- V \cup {p*} ; L\V <- L\V \ p*
    visited_vertices.insert(p_star);

    // L <- L \cup Nout(p*)  ; L \ V <- L \ V \cup Nout(p*)
    size_t bt{0};
    for (auto&& [_, p] : graph.out_edges(p_star)) {
      if (bt == L) {
        break;
      }

      // assert(p != p_star);
      if (!visited(p)) {
        auto score = distance(db[p], query);
        if constexpr (SP == SearchPath::path_and_search) {
          result.template insert(score, p);
        }
        q1.template insert(score, p);
        ++bt;
      }
    }
  }

  if constexpr (SP == SearchPath::path_and_search) {
    get_top_k_from_heap(result, nbd, k);
  }
  return visited_vertices;
}

template <class I = size_t, class Distance = sum_of_squares_distance>
auto greedy_path(
    auto&& graph,
    auto&& db,
    I source,
    auto&& query,
    size_t L,
    Distance&& distance = Distance{}) {
  return greedy_search<SearchPath::path_only>(graph, db, source, query, 1, L, std::vector<I>(1), distance);
}

template <class I = size_t, class Distance = sum_of_squares_distance>
auto greedy_search(
    auto&& graph,
    auto&& db,
    I source,
    auto&& query,
    size_t k,
    size_t L,
    auto&& nbd,
    Distance&& distance = Distance{}) {
  return greedy_search<SearchPath::path_and_search>(graph, db, source, query, k, L, nbd, distance);
}

/**
 * @brief RobustPrune(p, vee, alpha, R)
 * @tparam I index type
 * @tparam Distance distance functor
 * @param graph Graph
 * @param p point \in P
 * @param V candidate set
 * @param alpha distance threshold >= 1
 * @param R Degree bound
 *
 * V <- (V \cup Nout(p) \ p
 * Nout(p) < 0
 * while (!V.empty()) {
 *  Find p* \in V with smallest distance to p
 *  Nout(p) <- Nout(p) \cup p*
 *  if size(Nout(p)) == R {
 *    break
 *  }
 *  for pp \in V {
 *  if alpha * distance(p*, pp) <= distance(p, pp) {
 *    remove pp from V
 *  }
 * }
 */
template <class I = size_t, class Distance = sum_of_squares_distance>
auto robust_prune(
    auto&& graph,
    auto&& db,
    I p,
    auto&& V_in,
    float alpha,
    size_t R,
    Distance&& distance = Distance{}) {
  using T = typename std::decay_t<decltype(graph)>::value_type;
//  static_assert(
//      std::is_same_v<T, typename std::decay_t<decltype(db)>::value_type>,
//      "graph and db must be of same type");
  auto distance_comparator = [](auto&& a, auto&& b) {
    return std::get<0>(a) < std::get<0>(b);
  };
  using element = std::tuple<T, I>;
  auto V = std::set<element, std::function<bool(element, element)>>(
      distance_comparator);

  for (auto&& v : V_in) {
    if (v != p) {
      auto score = distance(db[v], db[p]);
      V.emplace(score, v);
    }
  }

  // V <- (V \cup Nout(p) \ p
  for (auto&& [ss, pp] : graph.out_edges(p)) {
    assert(pp != p);
    assert(ss == distance(db[p], db[pp]));
    V.emplace(ss, pp);
  }

  // Nout(p) <- 0
  graph.out_edges(p).clear();

  // while V != 0
  while (!V.empty()) {
    // p* <- argmin_{pp \in V} distance(p, pp)
    auto&& [s_star, p_star] = *(V.begin()); /* V.top(); */
    assert(p_star != p);

    // Nout(p) <- Nout(p) \cup p*
    graph.add_edge(p, p_star, s_star);

    if (graph.out_edges(p).size() == R) {
      break;
    }

    // For p' in V
    auto to_be_removed = std::vector<element>{};
    to_be_removed.reserve(V.size());
    for (auto&& [ss, pp] : V) {
      // if alpha * d(p*, p') <= d(p, p')
      if (alpha * distance(db[p_star], db[pp]) <= ss) {  // @todo <= or >=
        // V.erase({ss, pp});
        to_be_removed.emplace_back(ss, pp);
      }
    }
    for (auto&& [ss, pp] : to_be_removed) {
      V.erase({ss, pp});
    }
  }
}

template <class Distance = sum_of_squares_distance>
auto medioid(auto&& P, Distance distance = Distance{}) {
  auto n = P.num_cols();
  auto centroid = Vector<float>(P[0].size());
  for (size_t j = 0; j < P.num_cols(); ++j) {
    auto p = P[j];
    for (size_t i = 0; i < p.size(); ++i) {
      centroid[i] += p[i];
    }
  }
  for (size_t i = 0; i < centroid.size(); ++i) {
    centroid[i] /= P.num_cols();
  }
  auto min = std::numeric_limits<float>::max();
  auto med = 0UL;
  for (size_t i = 0; i < n; ++i) {
    auto score = distance(P[i], centroid);
    if (score < min) {
      min = score;
      med = i;
    }
  }
  return med;
}

template <class attribute_type, class shuffled_ids_type, class indices_type>
class vamana_index {
  std::unique_ptr<tdbColMajorMatrix<attribute_type>> feature_vectors_;

  size_t dimension_{0};
  size_t num_vectors_{0};
  size_t L_build_{0};       // diskANN paper says default = 100
  size_t R_max_degree_{0};  // diskANN paper says default = 64
  float alpha_min_ {1.0};   // per diskANN paper
  float alpha_max_ {1.2};   // per diskANN paper
  ::detail::graph::nn_graph<attribute_type, indices_type> graph_;

 public:
  vamana_index() = delete;
  vamana_index(const vamana_index& index) = delete;
  vamana_index& operator=(const vamana_index& index) = delete;
  vamana_index(vamana_index&& index) {
  }
  vamana_index& operator=(vamana_index&& index) = default;

  ~vamana_index() = default;

  vamana_index(size_t L, size_t R) : L_build_{L}, R_max_degree_{R} {
  }

  template <feature_vector_array V>
  void train(const V& training_set) {
    dimension_ = dimension(training_set);
    num_vectors_ = num_vectors(training_set);
    graph_ = ::detail::graph::init_random_nn_graph<float>(
        training_set, R_max_degree_);

    for (float alpha : {alpha_min_, alpha_max_}) {
      auto start = medioid(training_set);
      for (size_t p = 0; p < num_vectors(training_set); ++p) {
        auto visited =
            greedy_path(graph_, training_set, start, training_set[p], L_build_);
        robust_prune(graph_, training_set, p, visited, alpha, R_max_degree_);
        for (auto&& [i, j] : graph_.out_edges(p)) {
          if (graph_.out_degree(j) >= R_max_degree_) {
            robust_prune(
                graph_, training_set, j, training_set[j], alpha, R_max_degree_);
          }
        }
      }
    }
  }

  template <query_vector_array Q>
  void add(const Q& database){}

  template <query_vector_array Q>
  auto query(const Q& query_set, size_t k) {}

  auto remove(){}

  auto update(){}

  auto dimension(){}

  auto ntotal(){}

  auto num_vectors(){}
  
};



#endif  // TDB_VAMANA_H