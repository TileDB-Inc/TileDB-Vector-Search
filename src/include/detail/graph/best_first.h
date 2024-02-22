/**
 * @file   best_first.h
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

#ifndef TILEDB_BEST_FIRST_H
#define TILEDB_BEST_FIRST_H

#include <iostream>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

#include <queue>

#include "utils/minmax_and_dary_heap.h"

template <class Node>
struct compare_node {
  bool operator()(const Node& lhs, const Node& rhs) const {
    return std::get<0>(lhs) > std::get<0>(rhs);
  }
};

template <
    class Graph,
    feature_vector_array A,
    feature_vector V,
    class Distance = sum_of_squares_distance>
auto best_first_O0(
    const Graph& graph,
    const A& db,
    typename std::decay_t<Graph>::id_type source,
    const V& query,
    Distance&& distance = Distance{}) {
  using id_type = typename std::decay_t<Graph>::id_type;
  using score_type = float;
  using node_type = std::tuple<score_type, id_type>;

  std::
      priority_queue<node_type, std::vector<node_type>, compare_node<node_type>>
          pq;
  std::unordered_set<id_type> visited;

  pq.push({distance(db[source], query), source});

  while (!pq.empty()) {
    auto&& [_, p_star] = pq.top();
    pq.pop();

    // Explore successor nodes
    for (auto&& [_, neighbor_id] : graph[p_star]) {
      if (visited.count(neighbor_id) > 0)
        continue;

      score_type heuristic = distance(db[neighbor_id], query);
      if (heuristic == 0) {
        std::cout << "Found goal!" << std::endl;
        return true;
      }
      pq.push({heuristic, neighbor_id});
      visited.insert(neighbor_id);
    }
  }

  std::cout << "Goal not found: visited " << size(visited) << std::endl;
  return false;
}

template <
    class Graph,
    feature_vector_array A,
    feature_vector V,
    class Distance = sum_of_squares_distance>
auto best_first_O1(
    const Graph& graph,
    const A& db,
    typename std::decay_t<Graph>::id_type source,
    const V& query,
    size_t Lmax,
    Distance&& distance = Distance{}) {
  using id_type = typename std::decay_t<Graph>::id_type;
  using score_type = float;
  using node_type = std::tuple<score_type, id_type>;

  auto pq = k_min_heap<score_type, id_type>{Lmax};
  auto frontier = std::vector<node_type>{};
  std::unordered_set<id_type> visited;

  frontier.reserve(Lmax);
  pq.insert(distance(db[source], query), source);
  frontier.push_back({distance(db[source], query), source});

  while (!frontier.empty()) {
    auto&& [_, p_star] = frontier.back();
    frontier.pop_back();

    assert(!(visited.count(p_star) > 0));
    visited.insert(p_star);

    for (auto&& [_, neighbor_id] : graph[p_star]) {
      if (visited.count(neighbor_id) > 0)
        continue;

      score_type heuristic = distance(db[neighbor_id], query);
      frontier.push_back({heuristic, neighbor_id});
    }
  }

  std::cout << "Goal not found: visited " << size(visited) << std::endl;
  return false;
}


template <
    class Graph,
    feature_vector_array A,
    feature_vector V,
    class Distance = sum_of_squares_distance>
auto best_first_O2(
    const Graph& graph,
    const A& db,
    typename std::decay_t<Graph>::id_type source,
    const V& query,
    size_t Lmax,
    Distance&& distance = Distance{}) {
  using id_type = typename std::decay_t<Graph>::id_type;
  using score_type = float;
  using node_type = std::tuple<score_type, id_type>;

  auto pq = k_min_heap<score_type, id_type>{Lmax};
  auto frontier = std::vector<node_type>();
  std::unordered_set<id_type> visited;
  std::unordered_set<id_type> enfrontiered;
  std::unordered_set<id_type> enpqd;

  frontier.reserve(Lmax);
  score_type heuristic = distance(db[source], query);
  pq.insert(heuristic, source);
  enpqd.insert(source);

  frontier.emplace_back(heuristic, source);
  push_minmax_heap(frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
    return std::get<0>(a) < std::get<0>(b);
  });
  enfrontiered.insert(source);

  while (!frontier.empty()) {
    assert(is_minmax_heap(frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    }));

#if 0
    std::cout << "\n\n{\n";
    for (auto&& [ss, ii] : frontier) {
      std::cout << "{ " << ss << ", " << ii << "}, \n";
    }
    std::cout << "}\n";
#endif

    // Extract vertex with minimum score from frontier
    auto [_, p_star] = frontier.front();
    pop_minmax_heap_min(
        frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
          return std::get<0>(a) < std::get<0>(b);
        });

#if 0
    std::cout << "\n\n{\n";
    for (auto&& [ss, ii] : frontier) {
      //print_types(ss,ii);
      std::cout << "{ " << ss << ", " << ii << "}, \n";
    }
    std::cout << "}\n";
#endif

    auto&&[__, q_star] = frontier.back();
    auto&&[___, r_star] = frontier.front();

    frontier.pop_back();
    enfrontiered.erase(p_star);

    assert(p_star == q_star);

    visited.insert(p_star);

    score_type min_evicted_score = std::numeric_limits<score_type>::max();
    auto evicted_pq = std::vector<node_type>{};

    for (auto&& [_, neighbor_id] : graph[p_star]) {
      if (visited.count(neighbor_id) > 0 ||
          enfrontiered.count(neighbor_id) > 0 || enpqd.count(neighbor_id) > 0)
        continue;

      score_type heuristic = distance(db[neighbor_id], query);

      auto&& [inserted, evicted, evicted_score, evicted_id] =
          pq.template evict_insert<unique_id>(heuristic, neighbor_id);

      if (inserted) {
        enpqd.insert(neighbor_id);

        frontier.push_back({heuristic, neighbor_id});
        push_minmax_heap(
            frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
              return std::get<0>(a) < std::get<0>(b);
            });

        enfrontiered.insert(neighbor_id);

        if (evicted) {
          enpqd.erase(evicted_id);
          evicted_pq.push_back({evicted_score, evicted_id});
          min_evicted_score = std::min(min_evicted_score, evicted_score);
        }
      } else {
        evicted_pq.push_back({evicted_score, evicted_id});
        min_evicted_score = std::min(min_evicted_score, evicted_score);
      }
    }

    while (size(frontier) > 0) {
      score_type score_3;
      id_type id_3;
      if (size(frontier) == 1) {
        score_3 = std::get<0>(frontier.front());
        id_3 = std::get<1>(frontier.front());
      } else if (size(frontier) == 2) {
        score_3 = std::get<0>(frontier[1]);
        id_3 = std::get<1>(frontier[1]);
      } else {
        score_type score_1 = std::get<0>(frontier[1]);
        score_type score_2 = std::get<0>(frontier[2]);
        if (score_1 > score_2) {
          score_3 = std::get<0>(frontier[1]);
          id_3 = std::get<1>(frontier[1]);
        } else {
          score_3 = std::get<0>(frontier[2]);
          id_3 = std::get<1>(frontier[2]);
        }
      }
      if (score_3 < min_evicted_score) {
        break;
      }

      if (enfrontiered.count(id_3) > 0) {
        pop_minmax_heap_max(frontier.begin(), frontier.end());
        frontier.pop_back();
        enfrontiered.erase(id_3);
      } else {
        break;
      }
    }
  }

  std::cout << "Goal not found: visited " << size(visited) << " " << size(pq)
            << std::endl;
  return false;
}

#endif  // TILEDB_BEST_FIRST_H
