/**
 * @file   bfs.h
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

#ifndef TILEDB_BFS_H
#define TILEDB_BFS_H

#include <algorithm>
#include <deque>
#include <iostream>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include "utils/fixed_min_heap.h"

template <
    class Graph,
    feature_vector_array A,
    feature_vector V,
    class Distance = sum_of_squares_distance>
auto bfs_O1(
    const Graph& graph,
    const A& db,
    typename std::decay_t<Graph>::id_type source,
    const V& query,
    size_t Lmax,
    Distance&& distance = Distance{}) {
  using vertex_id_type = typename std::decay_t<Graph>::id_type;
  using score_type = float;

  // std::deque<vertex_id_type> q1, q2;
  auto q1 = k_min_heap<score_type, vertex_id_type>{Lmax};
  auto q2 = k_min_heap<score_type, vertex_id_type>{Lmax};

  std::vector<vertex_id_type> level(
      graph.num_vertices(), std::numeric_limits<vertex_id_type>::max());
  std::vector<vertex_id_type> parents(
      graph.num_vertices(), std::numeric_limits<vertex_id_type>::max());
  size_t lvl = 0;

  q1.insert(distance(db[source], query), source);
  level[source] = lvl++;
  parents[source] = source;

  // We want the frontier to be the Lbuild closest points to the query, minus
  // any points that have already been visited
  while (!q1.empty()) {
    std::for_each(q1.begin(), q1.end(), [&](auto&& e) {
      auto&& [s, u] = e;
      std::for_each(graph[u].begin(), graph[u].end(), [&, u = u](auto&& x) {
        vertex_id_type v = std::get<1>(x);
        if (level[v] == std::numeric_limits<vertex_id_type>::max()) {
          q2.insert(distance(db[v], query), v);
          level[v] = lvl;
          parents[v] = u;
        }
      });
    });
    std::swap(q1, q2);
    q2.clear();
    ++lvl;
  }

  // Check for number visited vertices, used for debugging
  [[maybe_unused]] size_t counter = 0;
  for (auto&& p : parents) {
    if (p != std::numeric_limits<vertex_id_type>::max()) {
      ++counter;
    }
  }

  return parents;
}

template <class Graph>
auto bfs_O0(const Graph& graph, typename std::decay_t<Graph>::id_type source) {
  using vertex_id_type = typename std::decay_t<Graph>::id_type;

  std::deque<vertex_id_type> q1, q2;
  std::vector<vertex_id_type> level(
      graph.num_vertices(), std::numeric_limits<vertex_id_type>::max());
  std::vector<vertex_id_type> parents(
      graph.num_vertices(), std::numeric_limits<vertex_id_type>::max());
  size_t lvl = 0;

  q1.push_back(source);
  level[source] = lvl++;
  parents[source] = source;

  while (!q1.empty()) {
    std::for_each(q1.begin(), q1.end(), [&](vertex_id_type u) {
      std::for_each(graph[u].begin(), graph[u].end(), [&](auto&& x) {
        vertex_id_type v = std::get<1>(x);
        if (level[v] == std::numeric_limits<vertex_id_type>::max()) {
          q2.push_back(v);
          level[v] = lvl;
          parents[v] = u;
        }
      });
    });
    std::swap(q1, q2);
    q2.clear();
    ++lvl;
  }
  size_t counter = 0;
  for (auto&& p : parents) {
    if (p != std::numeric_limits<vertex_id_type>::max()) {
      ++counter;
    }
  }
  std::cout << "visited: " << counter << " of " << size(parents) << std::endl;

  return parents;
}

#endif  // TILEDB_BFS_H
