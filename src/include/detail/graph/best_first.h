/**
 * @file   best_first.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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

    if (visited.count(p_star) > 0) {
      throw std::runtime_error("[best_first@best_first_O1] Vertex was visited");
    }
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
    if (!is_minmax_heap(
            frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
              return std::get<0>(a) < std::get<0>(b);
            })) {
      throw std::runtime_error(
          "[best_first@best_first_O2] Frontier is not a min-max heap");
    }

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

    if (p_star != q_star) {
      throw std::runtime_error("[best_first@best_first_O2] p_star != q_star");
    }

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
    enfrontiered.erase(p_star);
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

// constexpr static const uint8_t unvisited = 0;
constexpr static const uint8_t enfrontiered = 0;
constexpr static const uint8_t enpqd = 1;
constexpr static const uint8_t visited = 2;
constexpr static const uint8_t evicted = 4;
// constexpr static const uint8_t finished = 8;

auto set_unvisited(uint8_t& state) {
  state = 0;
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
auto set_evicted(uint8_t& state) {
  state |= evicted;
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
auto clear_evicted(uint8_t& state) {
  state &= ~evicted;
}

bool is_unvisited(uint8_t state) {
  return state == 0;
}
bool is_enfrontiered(uint8_t state) {
  return (state & enfrontiered) != 0;
}
bool is_enpqd(uint8_t state) {
  return (state & enpqd) != 0;
}
bool is_visited(uint8_t state) {
  return (state & visited) != 0;
}
bool is_evicted(uint8_t state) {
  return (state & evicted) != 0;
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
  // Key is vertex id, value is state
  std::unordered_map<id_type, uint8_t> vertex_state_map;

  // Set to keep track of which vertices have been visited
  std::unordered_set<id_type> visited;

  frontier.reserve(Lmax);
  score_type heuristic = distance(db[source], query);
  pq.insert(heuristic, source);
  auto&& [source_iter, success] =
      vertex_state_map.emplace(std::make_pair(source, 0));
  if (!success) {
    throw std::runtime_error(
        "[best_first@best_first_O3] Could not insert source into "
        "vertex_state_map");
  }

  set_enpqd(source_iter->second);

  frontier.emplace_back(heuristic, source);
  push_minmax_heap(frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
    return std::get<0>(a) < std::get<0>(b);
  });
  set_enfrontiered(source_iter->second);
  set_enpqd(source_iter->second);

  while (!frontier.empty()) {
    if (!is_minmax_heap(
            frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
              return std::get<0>(a) < std::get<0>(b);
            })) {
      throw std::runtime_error(
          "[best_first@best_first_O3] Frontier is not a min-max heap");
    }

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

    auto p_star_iter = vertex_state_map.find(p_star);
    if (p_star_iter == vertex_state_map.end()) {
      throw std::runtime_error(
          "[best_first@best_first_O3] p_star_iter == vertex_state_map.end()");
    }

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

    if (p_star != q_star) {
      throw std::runtime_error("[best_first@best_first_O3] p_star != q_star");
    }

    score_type min_evicted_score = std::numeric_limits<score_type>::max();
    auto evicted_pq = std::vector<node_type>{};

    for (auto&& [_, neighbor_id] : graph[p_star]) {
      auto neighbor_state_iter = vertex_state_map.find(neighbor_id);

      if (neighbor_state_iter == vertex_state_map.end()) {
        auto [local_neighbor_state_iter, success] =
            vertex_state_map.emplace(neighbor_id, 0);
        if (!success) {
          throw std::runtime_error(
              "[best_first@best_first_O3] Could not insert neighbor_id into "
              "vertex_state_map");
        }
        neighbor_state_iter = local_neighbor_state_iter;
      } else {
        if (!is_unvisited(neighbor_state_iter->second)) {
          continue;
        }
      }
      auto debug_neighbor_state = neighbor_state_iter->second;

      score_type heuristic = distance(db[neighbor_id], query);

      // pq.template insert<unique_id>(heuristic, neighbor_id);
      auto [inserted, evicted, evicted_score, evicted_id] =
          pq.template evict_insert<unique_id>(heuristic, neighbor_id);
      if (inserted) {
        set_enpqd(neighbor_state_iter->second);
        if (evicted) {
          auto evicted_state_iter = vertex_state_map.find(evicted_id);
          if (evicted_state_iter == vertex_state_map.end()) {
            throw std::runtime_error(
                "[best_first@best_first_O3] evicted_state_iter == "
                "vertex_state_map.end()");
          }

          if (is_visited(evicted_state_iter->second)) {
            // Only mark as evicted if not visited
            set_evicted(evicted_state_iter->second);
          }
        }
      } else {
        set_evicted(neighbor_state_iter->second);
      }
    }

    // copy pq to frontier, minus any vertices that have been visited or evicted
    frontier.clear();
    for (size_t i = 0; i < size(pq); ++i) {
      auto id = std::get<1>(pq[i]);
      auto iter = vertex_state_map.find(id);
      if (!is_visited(iter->second) && !is_evicted(iter->second)) {
        frontier.push_back(pq[i]);
        set_enfrontiered(iter->second);
      }
    }
    make_minmax_heap(frontier.begin(), frontier.end(), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });

    if (size(frontier) > size(pq)) {
      throw std::runtime_error(
          "[best_first@best_first_O3] size(frontier) (" +
          std::to_string(size(frontier)) + ") > size(pq) (" +
          std::to_string(size(pq)) + ")");
    }
    // vertex_state_map.erase(p_star);
  }
  auto top_k = std::vector<id_type>(k_nn);
  auto top_k_scores = std::vector<score_type>(k_nn);

  get_top_k_with_scores_from_heap(pq, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(visited));
}

template <
    class Graph,
    feature_vector_array A,
    feature_vector V,
    class Distance = sum_of_squares_distance>
auto best_first_O4(
    const Graph& graph,
    const A& db,
    typename std::decay_t<Graph>::id_type source,
    const V& query,
    size_t k_nn,
    size_t Lmax,
    Distance&& distance = Distance{}) {
  using id_type = typename std::decay_t<Graph>::id_type;
  using score_type = float;

  auto pq = k_min_heap<score_type, id_type>{Lmax};

  // The frontier to hold the next vertices to explore.  We will use it as a
  // min-max heap.  It will never be larger than Lmax.
  //  auto frontier = std::vector<node_type>();

  // Don't actually need a frontier
  // auto frontier = std::vector<node_type>{Lmax};

  // Map to keep track of state of vertices
  // Key is vertex id, value is state
  // We don't use a map in O4, but instead tag the node ids
  std::vector<uint8_t> vertex_state_property_map(graph.num_vertices(), 0);

  // Set to keep track of which vertices have been visited
  // Will be returned from this function
  std::unordered_set<id_type> visited;

  scoped_timer __{tdb_func__};

  score_type heuristic = distance(db[source], query);
  pq.insert(heuristic, source);
  set_enpqd(vertex_state_property_map[source]);

  id_type p_star = source;
  set_enfrontiered(vertex_state_property_map[p_star]);

  do {
    visited.insert(p_star);
    set_visited(vertex_state_property_map[p_star]);

    for (auto&& [_, neighbor_id] : graph[p_star]) {
      auto debug_neighbor_id = neighbor_id;
      auto neighbor_state = vertex_state_property_map[neighbor_id];
      if (!is_unvisited(neighbor_state)) {
        continue;
      }
      set_enpqd(vertex_state_property_map[neighbor_id]);

      score_type heuristic = distance(db[neighbor_id], query);

      auto [inserted, evicted, evicted_score, evicted_id] =
          // pq.template evict_insert<unique_id>(heuristic, neighbor_id);
          pq.evict_insert(heuristic, neighbor_id);
      if (inserted) {
        set_enpqd(vertex_state_property_map[neighbor_id]);
        if (evicted) {
          // Only mark as evicted if not visited
          if (!is_visited(vertex_state_property_map[evicted_id])) {
            set_evicted(vertex_state_property_map[evicted_id]);
          }
          clear_enpqd(vertex_state_property_map[evicted_id]);
        }
      } else {
        set_evicted(vertex_state_property_map[neighbor_id]);
        clear_enpqd(vertex_state_property_map[neighbor_id]);
      }
    }
    clear_enfrontiered(vertex_state_property_map[p_star]);
    //    std::cout << "p_star " << p_star << std::endl;
    if (!is_visited(vertex_state_property_map[p_star])) {
      throw std::runtime_error(
          "[best_first@best_first_O4] p_star is not visited");
    }
    if (is_evicted(vertex_state_property_map[p_star])) {
      throw std::runtime_error("[best_first@best_first_O4] p_star is evicted");
    }
    if (is_enfrontiered(vertex_state_property_map[p_star])) {
      throw std::runtime_error(
          "[best_first@best_first_O4] p_star is enfrontiered");
    }

    p_star = std::numeric_limits<id_type>::max();
    auto p_min_score = std::numeric_limits<score_type>::max();

    for (auto&& [pq_score, pq_id] : pq) {
      if (pq_score < p_min_score) {
        auto pq_state = vertex_state_property_map[pq_id];

        if (is_evicted(pq_state)) {
          throw std::runtime_error(
              "[best_first@best_first_O4] pq_state is evicted");
        }
        if (!is_enpqd(pq_state)) {
          throw std::runtime_error(
              "[best_first@best_first_O4] pq_state is not enpqd");
        }
        if (!is_visited(pq_state)) {
          p_star = pq_id;
          p_min_score = pq_score;
        }
      }
    }

    // set_finished(vertex_state_property_map[p_star]);
  } while (p_star != std::numeric_limits<id_type>::max());

  auto top_k = std::vector<id_type>(k_nn);
  auto top_k_scores = std::vector<score_type>(k_nn);

  get_top_k_with_scores_from_heap(pq, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(visited));
}

template <
    class Graph,
    feature_vector_array A,
    feature_vector V,
    class Distance = sum_of_squares_distance>
auto best_first_O5(
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

  auto pq = k_min_heap<score_type, id_type>{Lmax};

  // Map to keep track of state of vertices
  // Key is vertex id, value is state
  std::unordered_map<id_type, uint8_t> vertex_state_map;

  // Set to keep track of which vertices have been visited
  std::unordered_set<id_type> visited;

  score_type heuristic = distance(db[source], query);
  pq.insert(heuristic, source);
  auto&& [source_iter, success] =
      vertex_state_map.emplace(std::make_pair(source, 0));
  if (!success) {
    throw std::runtime_error(
        "[best_first@best_first_O3] Could not insert source into "
        "vertex_state_map");
  }

  set_enpqd(source_iter->second);

  auto p_star = source;
  set_enfrontiered(vertex_state_map[p_star]);
  do {
    visited.insert(p_star);

    auto p_star_iter = vertex_state_map.find(p_star);
    if (p_star_iter == vertex_state_map.end()) {
      throw std::runtime_error(
          "[best_first@best_first_O3] p_star_iter == vertex_state_map.end()");
    }
    clear_enfrontiered(p_star_iter->second);
    set_visited(p_star_iter->second);

    for (auto&& [_, neighbor_id] : graph[p_star]) {
      auto neighbor_state_iter = vertex_state_map.find(neighbor_id);

      if (neighbor_state_iter == vertex_state_map.end()) {
        auto [local_neighbor_state_iter, success] =
            vertex_state_map.emplace(neighbor_id, 0);
        if (!success) {
          throw std::runtime_error(
              "[best_first@best_first_O3] Could not insert neighbor_id into "
              "vertex_state_map");
        }
        neighbor_state_iter = local_neighbor_state_iter;
      } else {
        if (!is_unvisited(neighbor_state_iter->second)) {
          continue;
        }
      }
      auto debug_neighbor_state = neighbor_state_iter->second;

      score_type heuristic = distance(db[neighbor_id], query);

      // pq.template insert<unique_id>(heuristic, neighbor_id);
      auto [inserted, evicted, evicted_score, evicted_id] =
          pq.template evict_insert<unique_id>(heuristic, neighbor_id);
      if (inserted) {
        set_enpqd(neighbor_state_iter->second);
        if (evicted) {
          auto evicted_state_iter = vertex_state_map.find(evicted_id);
          if (evicted_state_iter == vertex_state_map.end()) {
            throw std::runtime_error(
                "[best_first@best_first_O3] evicted_state_iter == "
                "vertex_state_map.end()");
          }
          set_evicted(evicted_state_iter->second);
        }
      } else {
        set_evicted(neighbor_state_iter->second);
      }
    }

    p_star = std::numeric_limits<id_type>::max();
    auto p_min_score = std::numeric_limits<score_type>::max();

    for (auto&& [pq_score, pq_id] : pq) {
      auto pq_state_iter = vertex_state_map.find(pq_id);

      if (is_evicted(pq_state_iter->second)) {
        throw std::runtime_error(
            "[best_first@best_first_O3] pq_state_iter is evicted");
      }
      if (!is_enpqd(pq_state_iter->second)) {
        throw std::runtime_error(
            "[best_first@best_first_O3] pq_state_iter is not enpqd");
      }

      if (!is_visited(pq_state_iter->second)) {
        if (pq_score < p_min_score) {
          p_star = pq_id;
          p_min_score = pq_score;
        }
      }
    }

  } while (p_star != std::numeric_limits<id_type>::max());

  auto top_k = std::vector<id_type>(k_nn);
  auto top_k_scores = std::vector<score_type>(k_nn);

  get_top_k_with_scores_from_heap(pq, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(visited));
}

/*
 * The main bottlenecks (per vtune) are:
 *   - Keeping vertex state
 *   - Maintaining list of nearest neighbors (pq, aka Ell)
 *
 * Other things to try:
 *   - Profile with vtune to find actual bottlenecks
 *   - Use AVX2 instructions to compute distances (use compiler to generate
 *     rather than trying to use intrinsics)
 *   - Use an unordered_set to keep node state and use upper bits of node id to
 *     store the state
 *   - Use a k_min_max_heap for pq so that min is easily found and max can be
 *     easily evicted when inserting
 *   - Use a vector for pq, emplace_back nodes as we go along and then use
 *     std::nth_element to limit size to Lmax, marking the vertexes past Lmax
 *     as evicted
 */

#endif  // TILEDB_BEST_FIRST_H
