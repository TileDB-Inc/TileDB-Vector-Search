/**
 * @file   greedy_search.h
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

#ifndef TILEDB_GREEDY_SEARCH_H
#define TILEDB_GREEDY_SEARCH_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <ranges>
#include <unordered_set>
#include <vector>

#include "scoring.h"
#include "utils/fixed_min_heap.h"

static inline bool noisy = false;
static inline bool noisy_robust_prune = false;
[[maybe_unused]] static inline void set_noisy(bool b) {
  noisy = b;
}
[[maybe_unused]] static inline void set_noisy_robust_prune(bool b) {
  noisy_robust_prune = b;
}

/**
 * @brief Developmental code for performance analysis / optimization of
 * greedy search. "O1" is the OG.
 */
template </* SearchPath SP, */ class Distance = sum_of_squares_distance>
auto greedy_search_O2(
    auto&& graph,
    auto&& db,
    typename std::decay_t<decltype(graph)>::id_type source,
    auto&& query,
    size_t k_nn,
    size_t Lmax,
    Distance&& distance = Distance{}) {
  std::unordered_set<typename std::decay_t<decltype(graph)>::id_type> L;
  std::unordered_set<typename std::decay_t<decltype(graph)>::id_type> V;
  std::unordered_set<typename std::decay_t<decltype(graph)>::id_type> L_minus_V;

  L.insert(source);
  L_minus_V.insert(source);

  size_t counter{0};

  while (!empty(L_minus_V)) {
    if (noisy) {
      std::cout << "\n:::: " << counter++ << " ::::" << std::endl;
    }

    auto p_star = std::min_element(
        begin(L_minus_V), end(L_minus_V), [&](auto&& a, auto&& b) {
          return distance(db[a], query) < distance(db[b], query);
        });
    for (auto&& p : graph.out_edges(*p_star)) {
      L.insert(std::get<1>(p));
      L_minus_V.insert(std::get<1>(p));
    }
    V.insert(*p_star);

    if (size(L) > Lmax) {
      // trim L to Lmax closest points to query
      std::vector<
          std::pair<typename std::decay_t<decltype(graph)>::id_type, float>>
          L_with_scores;
      for (auto&& p : L) {
        L_with_scores.emplace_back(p, distance(db[p], query));
      }
      std::sort(
          begin(L_with_scores), end(L_with_scores), [](auto&& a, auto&& b) {
            return a.second < b.second;
          });
      L.clear();
      for (size_t i = 0; i < Lmax && i < size(L_with_scores); ++i) {
        L.insert(L_with_scores[i].first);
      }
    }
    L_minus_V.clear();
    for (auto&& p : L) {
      L_minus_V.insert(p);
    }
    for (auto&& p : V) {
      L_minus_V.erase(p);
    }

    if (noisy) {
      std::cout << "L: ";
      for (auto&& p : L) {
        std::cout << "(" << p << " " << distance(db[p], query) << ") ";
      }
      std::cout << std::endl;
      std::cout << "V: ";
      for (auto&& p : V) {
        std::cout << "(" << p << " " << distance(db[p], query) << ") ";
      }
      std::cout << std::endl;
      std::cout << "L\\V: ";
      for (auto&& p : L_minus_V) {
        std::cout << "(" << p << " " << distance(db[p], query) << ") ";
      }
      std::cout << std::endl;
    }
  }
  auto min_scores = fixed_min_pair_heap<
      float,
      typename std::decay_t<decltype(graph)>::id_type>(k_nn);
  for (auto&& p : L) {
    min_scores.insert(distance(db[p], query), p);
  }
  auto top_k =
      std::vector<typename std::decay_t<decltype(graph)>::id_type>(k_nn);
  auto top_k_scores = std::vector<float>(k_nn);

  get_top_k_with_scores_from_heap(min_scores, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(V));

  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(V));
}

/**
 * @brief Naive implementation of greedy search, useful for debugging and
 * testing.  "O1" is the OG.
 * @tparam Distance distance functor
 * @param graph graph to be searched
 * @param db input vectors
 * @param source vertex to begin search
 * @param query destination vertex
 * @param k_nn number of neighbors to return
 * @param Lmax maximum size of frontier
 * @param distance distance functor
 * @return top_k_scores, top_k, visited vertices
 */
template </* SearchPath SP, */ class Distance = sum_of_squares_distance>
auto greedy_search_O0(
    auto&& graph,
    auto&& db,
    typename std::decay_t<decltype(graph)>::id_type source,
    auto&& query,
    size_t k_nn,
    size_t Lmax,
    Distance&& distance = Distance{}) {
  std::unordered_set<typename std::decay_t<decltype(graph)>::id_type> L;
  std::unordered_set<typename std::decay_t<decltype(graph)>::id_type> V;
  std::unordered_set<typename std::decay_t<decltype(graph)>::id_type> L_minus_V;

  L.insert(source);
  L_minus_V.insert(source);

  size_t counter{0};

  while (!empty(L_minus_V)) {
    if (noisy) {
      std::cout << "\n:::: " << counter++ << " ::::" << std::endl;
    }

    auto p_star = std::min_element(
        begin(L_minus_V), end(L_minus_V), [&](auto&& a, auto&& b) {
          return distance(db[a], query) < distance(db[b], query);
        });
    for (auto&& p : graph.out_edges(*p_star)) {
      L.insert(std::get<1>(p));
      L_minus_V.insert(std::get<1>(p));
    }
    V.insert(*p_star);

    if (size(L) > Lmax) {
      // trim L to Lmax closest points to query
      std::vector<
          std::pair<typename std::decay_t<decltype(graph)>::id_type, float>>
          L_with_scores;
      for (auto&& p : L) {
        L_with_scores.emplace_back(p, distance(db[p], query));
      }
      std::sort(
          begin(L_with_scores), end(L_with_scores), [](auto&& a, auto&& b) {
            return a.second < b.second;
          });
      L.clear();
      for (size_t i = 0; i < Lmax && i < size(L_with_scores); ++i) {
        L.insert(L_with_scores[i].first);
      }
    }
    L_minus_V.clear();
    for (auto&& p : L) {
      L_minus_V.insert(p);
    }
    for (auto&& p : V) {
      L_minus_V.erase(p);
    }

    if (noisy) {
      std::cout << "L: ";
      for (auto&& p : L) {
        std::cout << "(" << p << " " << distance(db[p], query) << ") ";
      }
      std::cout << std::endl;
      std::cout << "V: ";
      for (auto&& p : V) {
        std::cout << "(" << p << " " << distance(db[p], query) << ") ";
      }
      std::cout << std::endl;
      std::cout << "L\\V: ";
      for (auto&& p : L_minus_V) {
        std::cout << "(" << p << " " << distance(db[p], query) << ") ";
      }
      std::cout << std::endl;
    }
  }
  auto min_scores = fixed_min_pair_heap<
      float,
      typename std::decay_t<decltype(graph)>::id_type>(k_nn);
  for (auto&& p : L) {
    min_scores.insert(distance(db[p], query), p);
  }
  auto top_k =
      std::vector<typename std::decay_t<decltype(graph)>::id_type>(k_nn);
  auto top_k_scores = std::vector<float>(k_nn);

  get_top_k_with_scores_from_heap(min_scores, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(V));
}

/**
 * @brief Truncated best-first search
 * @tparam Distance The distance function used to compare vectors
 * @param graph Graph to be searched
 * @param source start node index
 * @param query query node index
 * @param k result size
 * @param L search list size, L >= k
 * @return result set ell containing k-approximate nearest neighbors to query
 * and set vee containing all the visited nodes
 *
 * Per the DiskANN paper, the algorithm is as follows:
 * 1. Initialize the result list with the source node and visited list with
 *    empty
 * 2. While the result list \ visited list is not empty
 *    a. Find p* in the result list \ visited list with the smallest distance to
 *       the query
 *    b. update the result list with the out neighbors of p*
 *    c. Add p* to the visited list d. If size of the result list > L, trim
 *      the result list to keep L closest points to query
 * 3. Copy the result list to the output
 *
 * This is essentially a best-first search with a fixed size priority queue
 *
 * @todo -- add a `SearchPath `template parameter to determine whether to
 * return the top k results of the search or just the path taken.
 * @todo -- remove printf debugging code
 * @todo -- would it be more efficient somehow to process multiple queries?
 */
template </* SearchPath SP, */ class Distance = sum_of_squares_distance>
auto greedy_search_O1(
    auto&& graph,
    auto&& db,
    typename std::decay_t<decltype(graph)>::id_type source,
    auto&& query,
    size_t k_nn,
    size_t L,
    Distance&& distance = Distance{},
    bool convert_to_db_ids = false) {
  // using feature_type = typename std::decay_t<decltype(graph)>::feature_type;
  using id_type = typename std::decay_t<decltype(graph)>::id_type;
  using score_type = typename std::decay_t<decltype(graph)>::score_type;

  static_assert(std::integral<id_type>);

  if (L < k_nn) {
    throw std::runtime_error(
        "[greedy_search_O1] L (" + std::to_string(L) + ") < k_nn (" +
        std::to_string(k_nn) + ")");
  }

  std::unordered_set<id_type> visited_vertices;
  auto visited = [&visited_vertices](auto&& v) {
    return visited_vertices.contains(v);
  };

  auto result = k_min_heap<score_type, id_type>{L};  // 𝓛: |𝓛| <= L
  // auto result = std::set<id_type>{};
  auto q1 = k_min_heap<score_type, id_type>{L};  // 𝓛 \ 𝓥
  auto q2 = k_min_heap<score_type, id_type>{L};  // 𝓛 \ 𝓥

  scoped_timer __{tdb_func__};

  // 𝓛 <- {s} and 𝓥 <- ∅
  result.insert(distance(db[source], query), source);

  // q1 = 𝓛 \ 𝓥 = {s}
  q1.insert(distance(db[source], query), source);

  size_t counter{0};

  // while 𝓛 \ 𝓥 ≠ ∅
  while (!q1.empty()) {
    if (noisy) {
      std::cout << "\n:::: " << counter++ << " ::::" << std::endl;
      debug_min_heap(q1, "q1: ", 1);
    }

    // p* <- argmin_{p ∈ 𝓛 \ 𝓥} distance(p, q)

    // Although we use the name `k_min_heap` -- it actually stores a finite
    // number of elements in a max heap (we remove the max element
    // every time we have a smaller element to insert).  Since we are using
    // a `k_min_heap` for q1, to get and pop the min element, we have to
    // change it to a min heap, get the min element, and then change it back
    // to a max heap.
    // @todo -- There must be a better way of doing this

    // Change q1 into a min_heap
    std::make_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) > std::get<0>(b);
    });

    // Get and pop the min element
    std::pop_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) > std::get<0>(b);
    });

    auto [s_star, p_star] = q1.back();
    q1.pop_back();

    if (noisy) {
      std::cout << "p*: " << p_star
                << " --  distance = " << distance(db[p_star], query)
                << std::endl;
    }

    // Change back to max heap
    std::make_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });

    if (visited(p_star)) {
      continue;
    }

    // V <- V \cup {p*} ; L\V <- L\V \ p*
    visited_vertices.insert(p_star);

    if (noisy) {
      debug_vector(visited_vertices, "visited_vertices: ");
      debug_min_heap(graph.out_edges(p_star), "Nout(p*): ", 1);
    }

    // q2 <- L \ V
    // @todo Is there a better way to do this?  By somehow removing
    // elements from q1?  We can probably make the insertion into q2
    // more efficient -- have a batch interface to insert into q2
    for (auto&& [s, p] : result) {
      if (!visited(p)) {
        q2.insert(s, p);
      }
    }

    // L <- L \cup Nout(p*)  ; L \ V <- L \ V \cup Nout(p*)

    // In looking at profiling, a majority of time is spent in this loop,
    // in visited() and in result.insert()
    for (auto&& [_, p] : graph.out_edges(p_star)) {
      // assert(p != p_star);
      if (!visited(p)) {
        auto score = distance(db[p], query);

        // unique id or not does not seem to make a difference
        // @todo (actually it does, but shouldn't need it -- need to
        // investigate) if (result.template insert /*<unique_id>*/ (score, p)) {
        if (result.template insert<unique_id>(score, p)) {
          q2.template insert<unique_id>(score, p);
        }
      }
    }

    if (noisy) {
      debug_min_heap(result, "result, aka Ell: ", 1);
      debug_min_heap(result, "result, aka Ell: ", 0);
    }

    q1.swap(q2);
    q2.clear();
  }

  // auto top_k = Vector<id_type>(k_nn);
  // auto top_k_scores = Vector<score_type>(k_nn);
  auto top_k = std::vector<id_type>(k_nn);
  auto top_k_scores = std::vector<score_type>(k_nn);

  get_top_k_with_scores_from_heap(result, top_k, top_k_scores);

  // Optionally convert from the vector indexes to the db IDs. Used during
  // querying to map to external IDs.
  if (convert_to_db_ids) {
    for (int i = 0; i < k_nn; ++i) {
      if (top_k[i] != std::numeric_limits<id_type>::max()) {
        top_k[i] = db.ids()[top_k[i]];
      }
    }
  }

  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(visited_vertices));
}

template </* SearchPath SP, */ class Distance = sum_of_squares_distance>
auto greedy_search(
    auto&& graph,
    auto&& db,
    typename std::decay_t<decltype(graph)>::id_type source,
    auto&& query,
    size_t k_nn,
    size_t L,
    Distance&& distance = Distance{},
    bool convert_to_db_ids = false) {
  if (graph.num_vertices() == 0) {
    using id_type = typename std::decay_t<decltype(graph)>::id_type;
    using score_type = typename std::decay_t<decltype(graph)>::score_type;
    auto top_k =
        std::vector<id_type>(k_nn, std::numeric_limits<id_type>::max());
    auto top_k_scores =
        std::vector<score_type>(k_nn, std::numeric_limits<score_type>::max());
    std::unordered_set<id_type> visited_vertices;
    return std::make_tuple(
        std::move(top_k_scores), std::move(top_k), std::move(visited_vertices));
  }

  return greedy_search_O1(
      std::forward<decltype(graph)>(graph),
      std::forward<decltype(db)>(db),
      source,
      std::forward<decltype(query)>(query),
      k_nn,
      L,
      std::forward<Distance>(distance),
      convert_to_db_ids);
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
 * From the DiskANN paper:
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
  // using feature_type = typename std::decay_t<decltype(graph)>::feature_type;
  using id_type = typename std::decay_t<decltype(graph)>::id_type;
  using score_type = typename std::decay_t<decltype(graph)>::score_type;

  std::unordered_map<id_type, score_type> V_map;

  for (auto&& v : V_in) {
    if (v != p) {
      auto score = distance(db[v], db[p]);
      V_map.try_emplace(v, score);
    }
  }

  // V <- (V \cup Nout(p) \ p
  for (auto&& [ss, pp] : graph.out_edges(p)) {
    // assert(pp != p);
    if (pp != p) {
      // assert(ss == distance(db[p], db[pp]));
      V_map.try_emplace(pp, ss);
    }
  }

  std::vector<std::tuple<score_type, id_type>> V;
  V.reserve(V_map.size() + R);
  std::vector<std::tuple<score_type, id_type>> new_V;
  new_V.reserve(V_map.size() + R);

  for (auto&& v : V_map) {
    V.emplace_back(v.second, v.first);
  }

  if (noisy_robust_prune) {
    debug_min_heap(V, "V: ", 1);
  }

  // Nout(p) <- 0
  graph.out_edges(p).clear();

  size_t counter{0};
  // while V != 0
  while (!V.empty()) {
    if (noisy_robust_prune) {
      std::cout << "\n:::: " << counter++ << " ::::" << std::endl;
    }

    // p* <- argmin_{pp \in V} distance(p, pp)
    auto&& [s_star, p_star] =
        *(std::min_element(begin(V), end(V), [](auto&& a, auto&& b) {
          return std::get<0>(a) < std::get<0>(b);
        }));

    if (p_star == p) {
      throw std::runtime_error("[robust_prune] p_star == p");
    }

    if (noisy_robust_prune) {
      std::cout << "::::" << p_star << std::endl;
      debug_min_heap(V, "V: ", 1);
    }

    // Nout(p) <- Nout(p) \cup p*
    graph.add_edge(p, p_star, s_star);

    if (noisy_robust_prune) {
      debug_min_heap(graph.out_edges(p), "Nout(p): ", 1);
    }

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
    if (noisy_robust_prune) {
      debug_min_heap(V, "after prune V: ", 1);
    }

    std::swap(V, new_V);
    new_V.clear();
  }
}

#endif  // TILEDB_GREEDY_SEARCH_H
