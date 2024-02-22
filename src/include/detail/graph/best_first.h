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
    size_t k_nn,
    size_t Lmax,
    Distance&& distance = Distance{}) {
  scoped_timer __{tdb_func__};

  using id_type = typename std::decay_t<Graph>::id_type;
  using score_type = float;
  using node_type = std::tuple<score_type, id_type>;

  auto pq = k_min_heap<score_type, id_type>{Lmax};
  auto frontier = std::vector<node_type>();

  // Unordered set to keep track of state of vertices
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
    assert(is_minmax_heap(
        frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
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

    auto&& [__, q_star] = frontier.back();
    auto&& [___, r_star] = frontier.front();

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

      // Making the unique_id is a bit expensive -- may want to see if there
      // is another way to do this
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
  auto top_k = std::vector<id_type>(k_nn);
  auto top_k_scores = std::vector<score_type>(k_nn);

  get_top_k_with_scores_from_heap(pq, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(visited));
}

// There should not be any finished nor any unvisited -- those are just not in
// the map enum class vertex_state { unvisited, enfrontiered, enpqd, visited,
// finished };

constexpr static const uint8_t unvisited = 0;
constexpr static const uint8_t enfrontiered = 1;
constexpr static const uint8_t enpqd = 2;
constexpr static const uint8_t visited = 4;
constexpr static const uint8_t finished = 8;

auto set_unvisited(uint8_t& state) {
  state |= unvisited;
}
auto set_enfrontiered(uint8_t& state) {
  state |= enfrontiered;
}
auto set_enpqd(uint8_t& state) {
  state |= enpqd;
}
auto set_visited(uint8_t& state) {
  state |= visited;
}
auto set_finished(uint8_t& state) {
  state |= finished;
}
auto clear_unvisited(uint8_t& state) {
  state &= ~unvisited;
}
auto clear_enfrontiered(uint8_t& state) {
  state &= ~enfrontiered;
}
auto clear_enpqd(uint8_t& state) {
  state &= ~enpqd;
}
auto clear_visited(uint8_t& state) {
  state &= ~visited;
}
auto clear_finished(uint8_t& state) {
  state &= ~finished;
}
auto is_unvisited(uint8_t state) {
  return state & unvisited;
}
auto is_enfrontiered(uint8_t state) {
  return state & enfrontiered;
}
auto is_enpqd(uint8_t state) {
  return state & enpqd;
}
auto is_visited(uint8_t state) {
  return state & visited;
}
auto is_finished(uint8_t state) {
  return state & finished;
}

template <
    class Graph,
    feature_vector_array A,
    feature_vector V,
    class Distance = sum_of_squares_distance>
auto best_first_O3(
    const Graph& graph,
    const A& db,
    typename std::decay_t<Graph>::id_type source,
    const V& query,
    size_t k_nn,
    size_t Lmax,
    Distance&& distance = Distance{}) {
  scoped_timer __{tdb_func__};

  using id_type = typename std::decay_t<Graph>::id_type;
  using score_type = float;
  using node_type = std::tuple<score_type, id_type>;

  auto pq = k_min_heap<score_type, id_type>{Lmax};
  auto frontier = std::vector<node_type>();

  // Map to keep track of state of vertices
  std::unordered_map<id_type, uint8_t> vertex_state_map;
  std::unordered_set<id_type> visited;

  frontier.reserve(Lmax);
  score_type heuristic = distance(db[source], query);
  pq.insert(heuristic, source);
  auto&& [source_iter, success] = vertex_state_map.emplace(std::make_pair(source, 0));

  // enpqd.insert(source);
  set_enpqd(source_iter->second);

  frontier.emplace_back(heuristic, source);
  push_minmax_heap(frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  // enfrontiered.insert(source);
  set_enfrontiered(source_iter->second);

  while (!frontier.empty()) {
    assert(is_minmax_heap(
        frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
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

    auto debug_p_star = p_star;

    // assert(vertex_state_map.find(p_star) == vertex_state_map.end());
    auto p_star_iter = vertex_state_map.find(p_star);
    assert(p_star_iter != vertex_state_map.end());

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

    auto&& [__, q_star] = frontier.back();
    auto&& [___, r_star] = frontier.front();

    frontier.pop_back();
    // auto p_star_iter = vertex_state_map.find(p_star);



    // enfrontiered.erase(p_star);
    visited.insert(p_star);
    clear_enfrontiered(p_star_iter->second);
    set_visited(p_star_iter->second);

    assert(p_star == q_star);

    score_type min_evicted_score = std::numeric_limits<score_type>::max();
    auto evicted_pq = std::vector<node_type>{};

    for (auto&& [_, neighbor_id] : graph[p_star]) {
      auto neighbor_state_iter = vertex_state_map.find(neighbor_id);

      if (neighbor_state_iter == vertex_state_map.end()) {
        auto [local_neighbor_state_iter, success] = vertex_state_map.emplace(std::make_pair(neighbor_id, 0));
        assert(success);
        neighbor_state_iter = local_neighbor_state_iter;
      } else {
        if (!is_unvisited(neighbor_state_iter->second)) {
          continue;
        }
      }
      auto debug_neighbor_state = neighbor_state_iter->second;


      score_type heuristic = distance(db[neighbor_id], query);

      // Making the unique_id is a bit expensive -- may want to see if there
      // is another way to do this
      auto&& [inserted, evicted, evicted_score, evicted_id] =
          pq.template evict_insert<unique_id>(heuristic, neighbor_id);

      if (inserted) {
        // enpqd.insert(neighbor_id);
        set_enpqd(neighbor_state_iter->second);

        frontier.push_back({heuristic, neighbor_id});
        push_minmax_heap(
            frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
              return std::get<0>(a) < std::get<0>(b);
            });
        set_enfrontiered(neighbor_state_iter->second);

        if (evicted) {
          clear_enpqd(vertex_state_map[evicted_id]);
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

      if (is_enfrontiered(vertex_state_map[id_3])) {
        pop_minmax_heap_max(frontier.begin(), frontier.end());
        frontier.pop_back();
        clear_enfrontiered(vertex_state_map[id_3]);
      } else {
        break;
      }
    }
  }
  auto top_k = std::vector<id_type>(k_nn);
  auto top_k_scores = std::vector<score_type>(k_nn);

  get_top_k_with_scores_from_heap(pq, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(visited));
}

#endif  // TILEDB_BEST_FIRST_H
