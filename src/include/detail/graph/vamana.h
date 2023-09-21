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

#include "detail/graph/graph_utils.h"
#include "detail/graph/adj_list.h"

#include <tiledb/tiledb>
#include <tiledb/group_experimental.h>

namespace detail::graph {
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
template </* SearchPath SP, */ class Distance = sum_of_squares_distance>
auto greedy_search(
    auto&& graph,
    auto&& db,
    typename std::decay_t<decltype(graph)>::index_type source,
    auto&& query,
    size_t k_nn,
    size_t L,
    Distance&& distance = Distance{}) {
  constexpr bool noisy = false;

  using value_type = typename std::decay_t<decltype(graph)>::value_type;
  using index_type = typename std::decay_t<decltype(graph)>::index_type;

  assert(L >= k_nn);

  std::set<index_type> visited_vertices;
  auto visited = [&visited_vertices](auto&& v) {
    return visited_vertices.find(v) != visited_vertices.end();
  };

  auto result = k_min_heap<value_type, index_type>{L};  // Ell: |Ell| <= L
  auto q1 = k_min_heap<value_type, index_type>{L};      // Ell \ V

  // L <- {s} and V <- empty`
  result.insert(distance(db[source], query), source);

  // q1 = L \ V = {s}
  q1.insert(distance(db[source], query), source);

  size_t counter{0};

  // while L\V is not empty
  while (!q1.empty()) {
    if (noisy)
      std::cout << "\n:::: " << counter++ << " ::::" << std::endl;
    if (noisy)
      debug_min_heap(q1, "q1: ", 1);

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

    if (noisy)
      std::cout << "p*: " << p_star << std::endl;

    // Change back to max heap
    std::make_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });

    if (visited(p_star)) {
      continue;
    }

    // V <- V \cup {p*} ; L\V <- L\V \ p*
    visited_vertices.insert(p_star);

    if (noisy)
      debug_vector(visited_vertices, "visited_vertices: ");
    if (noisy)
      debug_min_heap(graph.out_edges(p_star), "Nout(p*): ", 1);

    // @todo -- needed?
    // q1.clear(); // Or remove newly visited

    // L <- L \cup Nout(p*)  ; L \ V <- L \ V \cup Nout(p*)
    for (auto&& [_, p] : graph.out_edges(p_star)) {
      // assert(p != p_star);
      if (!visited(p)) {
        auto score = distance(db[p], query);
        if (result.template insert<unique_id>(score, p)) {
          q1.template insert<unique_id>(score, p);
        }
      }
    }

    if (noisy)
      debug_min_heap(result, "result, aka Ell: ", 1);
    if (noisy)
      debug_min_heap(result, "result, aka Ell: ", 0);
  }

  // auto top_k = Vector<index_type>(k_nn);
  // auto top_k_scores = Vector<value_type>(k_nn);
  auto top_k = std::vector<index_type>(k_nn);
  auto top_k_scores = std::vector<value_type>(k_nn);

  get_top_k_with_scores_from_heap(result, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(visited_vertices));
}

#if 0
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
    Distance&& distance = Distance{}) {
  return greedy_search<SearchPath::path_and_search>(graph, db, source, query, k, L, distance);
}
#endif

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
  constexpr bool noisy = false;

  using value_type = typename std::decay_t<decltype(graph)>::value_type;
  using index_type = typename std::decay_t<decltype(graph)>::index_type;

  std::vector<std::tuple<value_type, index_type>> V;
  V.reserve(V_in.size() + graph.out_degree(p));
  std::vector<std::tuple<value_type, index_type>> new_V;
  new_V.reserve(V.size());

  for (auto&& v : V_in) {
    if (v != p) {
      auto score = distance(db[v], db[p]);
      V.emplace_back(score, v);
    }
  }

  // V <- (V \cup Nout(p) \ p
  for (auto&& [ss, pp] : graph.out_edges(p)) {
    // assert(pp != p);
    if (pp != p) {
      // assert(ss == distance(db[p], db[pp]));
      V.emplace_back(ss, pp);
    }
  }

  if (noisy)
    debug_min_heap(V, "V: ", 1);

  // Nout(p) <- 0
  graph.out_edges(p).clear();

  size_t counter{0};
  // while V != 0
  while (!V.empty()) {
    if (noisy)
      std::cout << "\n:::: " << counter++ << " ::::" << std::endl;

    // p* <- argmin_{pp \in V} distance(p, pp)
    auto&& [s_star, p_star] =
        *(std::min_element(begin(V), end(V), [](auto&& a, auto&& b) {
          return std::get<0>(a) < std::get<0>(b);
        }));
    assert(p_star != p);
    if (noisy)
      std::cout << "::::" << p_star << std::endl;
    if (noisy)
      debug_min_heap(V, "V: ", 1);

    // Nout(p) <- Nout(p) \cup p*
    graph.add_edge(p, p_star, s_star);

    if (noisy)
      debug_min_heap(graph.out_edges(p), "Nout(p): ", 1);

    if (graph.out_edges(p).size() == R) {
      break;
    }

    // For p' in V
    for (auto&& [ss, pp] : V) {
      // if alpha * d(p*, p') <= d(p, p')
      // assert(ss == distance(db[p], db[pp]));
      if (alpha * distance(db[p_star], db[pp]) <= ss) {
        // V.erase({ss, pp});
        ;
      } else {
        if (pp != p) {
          new_V.emplace_back(ss, pp);
        }
      }
    }
    if (noisy)
      debug_min_heap(V, "after prune V: ", 1);

    // print_types(V, new_V);

    std::swap(V, new_V);
    new_V.clear();
    // V.unfiltered_heapify();
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
  std::vector<float> tmp{begin(centroid), end(centroid)};
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

/**
 * @brief
 * @tparam Array
 */
template <feature_vector_array Array>
class vamana_index {
  // Array feature_vectors_;
  using index_type = typename Array::index_type;
  using feature_type = typename Array::value_type;
  using id_type = typename Array::index_type;
  using score_type = float;

  ColMajorMatrix<feature_type> feature_vectors_;

  uint64_t dimension_{0};
  uint64_t num_vectors_{0};
  uint64_t L_build_{0};       // diskANN paper says default = 100
  uint64_t R_max_degree_{0};  // diskANN paper says default = 64
  uint64_t B_backtrack_{0};   //
  float alpha_min_{1.0};      // per diskANN paper
  float alpha_max_{1.2};      // per diskANN paper
  ::detail::graph::adj_list<score_type, index_type> graph_;
  index_type medioid_{0};

 public:
  vamana_index() = delete;
  vamana_index(const vamana_index& index) = delete;
  vamana_index& operator=(const vamana_index& index) = delete;
  vamana_index(vamana_index&& index) {
  }
  vamana_index& operator=(vamana_index&& index) = default;

  ~vamana_index() = default;

  vamana_index(size_t num_nodes, size_t L, size_t R, size_t B = 0)
      : num_vectors_{num_nodes}
      , L_build_{L}
      , R_max_degree_{R}
      , B_backtrack_{B == 0 ? L_build_ : B}
      , graph_{num_vectors_} {
  }

  void train(const Array& training_set) {
    feature_vectors_ = std::move(ColMajorMatrix<feature_type>(
        _cpo::dimension(training_set), _cpo::num_vectors(training_set)));
    std::copy(
        training_set.data(),
        training_set.data() +
            _cpo::dimension(training_set) * _cpo::num_vectors(training_set),
        feature_vectors_.data());

    dimension_ = _cpo::dimension(feature_vectors_);
    num_vectors_ = _cpo::num_vectors(feature_vectors_);
    graph_ = ::detail::graph::init_random_adj_list<float>(
        feature_vectors_, R_max_degree_);

    // dump_edgelist("edges_" + std::to_string(0) + ".txt", graph_);

    medioid_ = medioid(feature_vectors_);

    size_t counter{0};
    for (float alpha : {alpha_min_, alpha_max_}) {
      for (size_t p = 0; p < num_vectors_; ++p) {
        ++counter;
        auto&& [top_k_scores, top_k, visited] = greedy_search(
            graph_,
            feature_vectors_,
            medioid_,
            feature_vectors_[p],
            1,
            L_build_);
        robust_prune(
            graph_, feature_vectors_, p, visited, alpha, R_max_degree_);
        for (auto&& [i, j] : graph_.out_edges(p)) {
          // @todo Do this without copying -- prune should take vector of tuples and p (it copies anyway)
          auto tmp = std::vector<size_t>(graph_.out_degree(j) + 1);
          tmp.push_back(p);
          for (auto&& [_, k] : graph_.out_edges(j)) {
            tmp.push_back(k);
          }

          if (size(tmp) > R_max_degree_) {
            robust_prune(
                graph_, feature_vectors_, j, tmp, alpha, R_max_degree_);
          } else {
            graph_.add_edge(
                j,
                p,
                sum_of_squares_distance()(
                    feature_vectors_[p], feature_vectors_[j]));
          }
        }
        if ((counter) % 10 == 0) {
          // dump_edgelist("edges_" + std::to_string(counter) + ".txt", graph_);
        }
      }
    }
  }

  template <feature_vector_array A>
  void add(const A& database) {
  }

  template <query_vector_array Q>
  auto query(const Q& query_set, size_t k) {
    auto top_k = ColMajorMatrix<size_t>(k, ::num_vectors(query_set));
    auto top_k_scores = ColMajorMatrix<float>(k, ::num_vectors(query_set));
    for (size_t i = 0; i < ::num_vectors(query_set); ++i) {
      auto&& [_top_k_scores, _top_k, V] = greedy_search(
          graph_, feature_vectors_, medioid_, query_set[i], k, L_build_);
      std::copy(
          _top_k_scores.data(),
          _top_k_scores.data() + k,
          top_k_scores[i].data());
      std::copy(_top_k.data(), _top_k.data() + k, top_k[i].data());
    }
    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  template <query_vector Q>
  auto query(const Q& query_vec, size_t k) {
    auto&& [top_k_scores, top_k, V] = greedy_search(
        graph_, feature_vectors_, medioid_, query_vec, k, L_build_);
    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  auto remove() {
  }

  auto update() {
  }

  constexpr auto dimension() const {
    return dimension_;
  }

  constexpr auto ntotal() const {
    return num_vectors_;
  }

  using metadata_element = std::tuple<std::string, void*, tiledb_datatype_t>;
  std::vector<metadata_element> metadata{
      {"dimension", &dimension_, TILEDB_UINT64},
      {"ntotal", &num_vectors_, TILEDB_UINT64},
      {"L", &L_build_, TILEDB_UINT64},
      {"R", &R_max_degree_, TILEDB_UINT64},
      {"B", &B_backtrack_, TILEDB_UINT64},
      {"alpha_min", &alpha_min_, TILEDB_FLOAT32},
      {"alpha_max", &alpha_max_, TILEDB_FLOAT32},
      {"medioid", &medioid_, TILEDB_UINT64},
  };

  template <class ValueType, class IndexType>
  auto write_index(const std::string& group_uri, bool overwrite = false) {
    // copilot ftw!
    // metadata: dimension, ntotal, L, R, B, alpha_min, alpha_max, medioid
    // Save as a group: metadata, feature_vectors, graph edges, offsets

    tiledb::Context ctx;
    tiledb::VFS vfs(ctx);
    if (vfs.is_dir(group_uri)) {
      if (overwrite == false) {
        return false;
      }
      vfs.remove_dir(group_uri);
    }

    tiledb::Config cfg;
    tiledb::Group::create(ctx, group_uri);
    auto write_group = tiledb::Group(ctx, group_uri, TILEDB_WRITE, cfg);

    for (auto&& [name, value, type] : metadata) {
      write_group.put_metadata(name, type, 1, value);
    }

    // feature_vectors
    auto feature_vectors_uri = group_uri + "/feature_vectors";
    write_matrix(ctx, feature_vectors_, feature_vectors_uri);
    write_group.add_member(feature_vectors_uri, true, "feature_vectors");

    // adj_list
    auto adj_scores_uri = group_uri + "/adj_scores";
    auto adj_ids_uri = group_uri + "/adj_ids";
    auto adj_index_uri = group_uri + "/adj_index";
    auto adj_scores = Vector<score_type> {graph_.num_edges()};
    auto adj_ids = Vector<id_type> {graph_.num_edges()};
    auto adj_index = Vector<uint64_t> (graph_.num_vertices() + 1);

    size_t edge_offset{0};
    for (size_t i = 0; i < num_vertices(graph_); ++i) {
      adj_index[i] = edge_offset;
      for (auto&& [score, id] : graph_.out_edges(i)) {
        adj_scores[edge_offset] = score;
        adj_ids[edge_offset] = id;
        ++edge_offset;
      }
    }
    adj_index.back() = edge_offset;

    write_vector(ctx, adj_scores, adj_scores_uri);
    write_group.add_member(adj_scores_uri, true, "adj_scores");

    write_vector(ctx, adj_ids, adj_ids_uri);
    write_group.add_member(adj_ids_uri, true, "adj_ids");

    write_vector(ctx, adj_index, adj_index_uri);
    write_group.add_member(adj_index_uri, true, "adj_index");

    write_group.close();
    return true;
  }
};

}  // namespace detail::graph
#endif  // TDB_VAMANA_H