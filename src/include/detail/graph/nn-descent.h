/**
 * @file   nn-descent.h
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

#ifndef TILEDB_NN_DESCENT_H
#define TILEDB_NN_DESCENT_H

#include <unordered_set>
#include "detail/graph/nn-graph.h"
#include "scoring.h"
#include "utils/print_types.h"

namespace detail::graph {
/*
 Basic nn-search algorithm

1. Choose a starting node in the graph (potentially randomly) as a candidate
node
2. Look at all nodes connected by an edge to the best untried candidate node in
the graph
3. Add all these nodes to our potential candidate pool
4. Sort the candidate pool by closeness / similarity to the query point
5. Truncate the pool to 𝑘 best (as in closest to the query) candidates
6. Return to step 2, unless we have already tried all the candidates in the pool

  */

/*
    NN-descent algorithm
    1. Start with a random graph (connect each node to 𝑘 random nodes)
    2. For each node:
    3. Measure the distance from the node to the neighbors of its neighbors
    4. If any are closer, then update the graph accordingly, and keep only the 𝑘
   closest
    5. If any updates were made to the graph then go back to step 2, otherwise
   stop
 */

template <
    std::integral I,
    std::integral J,
    class Distance = sum_of_squares_distance>
auto nn_descent_0_step(
    auto&& g, auto&& db, I i, J j, Distance distance = Distance()) {
  size_t num_updates{0};
  for (auto&& [_, k] : out_edges(g, j)) {  // _ is dist(j, k)
    if (k == i) {
      continue;
    }
    auto ik_score = distance(db[i], db[k]);
    if (g.add_edge(i, k, ik_score)) {
      ++num_updates;
    }
  }
  for (auto&& k : in_edges(g, j)) {  // _ is dist(j, k)
    if (k == i) {
      continue;
    }
    auto ik_score = distance(db[i], db[k]);
    if (g.add_edge(i, k, ik_score)) {
      ++num_updates;
    }
  }

  return num_updates;
}

template <class Distance = sum_of_squares_distance>
auto nn_descent_0_step_all(
    auto&& g, auto&& db, Distance distance = Distance()) {
  size_t num_updates{0};
  size_t nvertices{num_vertices(g)};
  // g.copy_to_update_edges();

  g.build_in_edges();

  for (size_t i = 0; i < nvertices; ++i) {
    for (auto&& [_, j] : out_edges(g, i)) {
      num_updates += nn_descent_0_step(g, db, i, j);
    }

    for (auto&& j : in_edges(g, i)) {
      num_updates += nn_descent_0_step(g, db, i, j);
    }
  }
  // g.swap_all_update_edges();

  return num_updates;
}

// Local join
template <
    std::integral I,
    std::integral J,
    class Distance = sum_of_squares_distance>
auto nn_descent_1_step(
    auto&& g, auto&& db, I i, J j, Distance distance = Distance()) {
  size_t num_updates{0};

  for (auto&& [_, k] : out_edges(g, i)) {  // _ is dist(j, k)
    if (k <= j) {
      continue;
    }
    auto jk_score = distance(db[j], db[k]);
    if (g.add_edge(j, k, jk_score)) {
      ++num_updates;
    }
    if (g.add_edge(k, j, jk_score)) {
      ++num_updates;
    }
  }

  for (auto&& k : in_edges(g, i)) {  // _ is dist(j, k)
    if (k <= j) {
      continue;
    }
    auto jk_score = distance(db[j], db[k]);
    if (g.add_edge(j, k, jk_score)) {
      ++num_updates;
    }
    if (g.add_edge(k, j, jk_score)) {
      ++num_updates;
    }
  }

  return num_updates;
}

template <class Distance = sum_of_squares_distance>
auto nn_descent_1_step_all(
    auto&& g, auto&& db, Distance distance = Distance()) {
  size_t num_updates{0};
  size_t nvertices{num_vertices(g)};
  // g.copy_to_update_edges();

  g.build_in_edges();

  for (size_t i = 0; i < nvertices; ++i) {
    for (auto&& [_, j] : out_edges(g, i)) {
      // print_types(i, j);
      num_updates += nn_descent_1_step(g, db, i, j);
    }

    for (auto&& j : in_edges(g, i)) {
      num_updates += nn_descent_1_step(g, db, i, j);
    }
  }
  // g.swap_all_update_edges();

  return num_updates;
}

#if 0
template <adjacency_list_graph Graph>
auto bfs(const Graph& graph, vertex_id_t<Graph> root) {
  using vertex_id_type = vertex_id_t<Graph>;

  std::deque<vertex_id_type>  q1, q2;
  std::vector<vertex_id_type> level(num_vertices(graph), std::numeric_limits<vertex_id_type>::max());
  std::vector<vertex_id_type> parents(num_vertices(graph), std::numeric_limits<vertex_id_type>::max());
  size_t                      lvl = 0;

  q1.push_back(root);
  level[root]   = lvl++;
  parents[root] = root;

  while (!q1.empty()) {

    std::for_each(q1.begin(), q1.end(), [&](vertex_id_type u) {
      std::for_each(graph[u].begin(), graph[u].end(), [&](auto&& x) {
        vertex_id_type v = target(graph, x);
        if (level[v] == std::numeric_limits<vertex_id_type>::max()) {
          q2.push_back(v);
          level[v]   = lvl;
          parents[v] = u;
        }
      });
    });
    std::swap(q1, q2);
    q2.clear();
    ++lvl;
  }
  return parents;
}

#endif

template <class Graph>
struct vertex_id {
  using type = typename Graph::vertex_id_type;
};

template <class Graph>
using vertex_id_t = typename vertex_id<Graph>::type;

template <class T, class I>
struct vertex_id<nn_graph<T, I>> {
  using type = I;
};

template <class T, class I>
auto bfs(nn_graph<T, I>& graph, I root) {
  using vertex_id_type = I;

  std::deque<vertex_id_type> q1, q2;
  std::vector<vertex_id_type> level(
      num_vertices(graph), std::numeric_limits<vertex_id_type>::max());
  std::vector<vertex_id_type> parents(
      num_vertices(graph), std::numeric_limits<vertex_id_type>::max());
  size_t lvl = 0;

  q1.push_back(root);
  level[root] = lvl++;
  parents[root] = root;

  while (!q1.empty()) {
    std::for_each(q1.begin(), q1.end(), [&](vertex_id_type u) {
      std::for_each(
          graph.out_edges(u).begin(), graph.out_edges(u).end(), [&](auto&& x) {
            auto&& [_, v] = x;
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

  for (auto&& l : level) {
    if (l == std::numeric_limits<vertex_id_type>::max()) {
      std::cout << "Graph is not connected!" << std::endl;
      break;
    }
  }

  return parents;
}

template <class T, std::integral ID, class Distance = sum_of_squares_distance>
auto nn_descent_1_query(
    const detail::graph::nn_graph<T, ID>& graph,
    auto&& db,
    auto&& query,
    size_t k_nn,
    size_t num_seeds,
    size_t nthreads,
    Distance distance = Distance()) {
  using vertex_id_type = ID;

  auto num_queries = query.num_cols();

  fixed_min_pair_heap<float, vertex_id_type> q1{k_nn}, q2{k_nn};
  auto top_k = std::vector<fixed_min_pair_heap<float, vertex_id_type>>(
      num_queries, fixed_min_pair_heap<float, vertex_id_type>(k_nn));

  for (size_t i = 0; i < num_queries; ++i) {
    q1.clear();
    q2.clear();
    auto q = query[i];

    std::vector<vertex_id_type> level(
        num_vertices(graph), std::numeric_limits<vertex_id_type>::max());

    for (size_t j = 0; j < num_seeds; ++j) {
      auto gen = std::mt19937_64(std::random_device()());
      auto root = std::uniform_int_distribution<size_t>(
          0, num_vertices(graph) - 1)(gen);
      q1.insert(distance(db[root], q), root);
      top_k[i].insert(distance(db[root], q), root);
      level[root] = 0;
    }
    size_t lvl = 1;

    while (!q1.empty()) {
      auto start = q1.begin();
      auto stop = q1.end();
      if (lvl != 0) {
        std::sort_heap(begin(q1), end(q1));
        stop = start + 1;
      }

      std::for_each(start, stop, [&](auto&& x) {
        auto&& [_, u] = x;
        //       auto&& [_, u] = q1.front(); // this is going to be max element

        auto nbd{graph.entire_neighborhood(u)};
        auto in_start = begin(nbd);
        auto in_stop = end(nbd);

        std::for_each(in_start, in_stop, [&](auto&& v) {
          if (level[v] == std::numeric_limits<vertex_id_type>::max()) {
            auto dist = distance(db[v], q);
            q2.template insert<unique_id>(dist, v);
            level[v] = lvl;
            top_k[i].insert(dist, v);
          }
        });
      });
      std::swap(q1, q2);
      q2.clear();
      ++lvl;
    }
  }
  return get_top_k_with_scores(top_k, k_nn);
}

template <class T, std::integral ID, class Distance = sum_of_squares_distance>
auto nn_descent_1(auto&& db, size_t k_nn, Distance distance = Distance()) {
  auto g = init_random_nn_graph<T, ID, Distance>(db, k_nn, distance);

  size_t num_updates{0};
  do {
    num_updates = nn_descent_1_step_all(g, db);
    std::cout << num_updates << std::endl;
  } while (num_updates > (k_nn * num_vertices(g)) / 100);

  return g;
}

template <std::integral I, std::integral J, std::integral ID>
auto nn_descent_step(auto&& g, I i, J j, std::unordered_set<ID>& s) {
  auto num_visited = 0;

  for (auto&& [_, k] : out_edges(g, j)) {  // _ is dist(j, k)
    if (i == k) {
      continue;
    }
    s.insert(k);
    ++num_visited;
  }
  for (auto&& k : in_edges(g, j)) {  // _ is dist(j, k)
    if (i == k) {
      continue;
    }
    s.insert(k);
    ++num_visited;
  }
  return num_visited;
}

template <class Distance = sum_of_squares_distance>
auto nn_descent_step_all(auto&& g, auto&& db, Distance distance = Distance()) {
  scoped_timer t("nn_descent_step_all");
  size_t num_candidates{0};
  size_t num_updates{0};
  size_t num_visited{0};
  size_t nvertices{num_vertices(g)};
  // g.copy_to_update_edges();

  g.build_in_edges();

  std::unordered_set<size_t> joined;

  for (size_t i = 0; i < nvertices; ++i) {
    joined.clear();
    for (auto&& [_, j] : out_edges(g, i)) {
      num_visited += nn_descent_step(g, i, j, joined);
    }
    for (auto&& j : in_edges(g, i)) {
      num_visited += nn_descent_step(g, i, j, joined);
    }
    std::for_each(begin(joined), end(joined), [&](auto&& k) {
      ++num_candidates;
      auto ik_score = distance(db[i], db[k]);
      if (g.add_edge(i, k, ik_score)) {
        ++num_updates;
      }
      if (g.add_edge(k, i, ik_score)) {
        ++num_updates;
      }
    });
  }

  t.stop();

  std::cout << num_visited << ", " << num_updates << ", " << num_candidates
            << std::endl;

  return num_updates;
}

template <class Distance = sum_of_squares_distance>
auto nn_descent_step_full_all(
    auto&& g, auto&& db, Distance distance = Distance()) {
  scoped_timer t("nn_descent_step_full_all");
  size_t num_candidates{0};
  size_t num_updates{0};
  size_t num_visited{0};
  size_t nvertices{num_vertices(g)};
  // g.copy_to_update_edges();

  g.build_in_edges();

  std::unordered_set<size_t> joined;

  for (size_t i = 0; i < nvertices; ++i) {
    joined.clear();
    for (auto&& [_, j] : out_edges(g, i)) {
      num_visited += nn_descent_step(g, i, j, joined);
    }
    for (auto&& j : in_edges(g, i)) {
      num_visited += nn_descent_step(g, i, j, joined);
    }
    std::for_each(begin(joined), end(joined), [&](auto&& k) {
      ++num_candidates;
      auto ik_score = distance(db[i], db[k]);
      if (g.add_edge(i, k, ik_score)) {
        ++num_updates;
      }
      if (g.add_edge(k, i, ik_score)) {
        ++num_updates;
      }
    });
  }

  t.stop();

  std::cout << num_visited << ", " << num_updates << ", " << num_candidates
            << std::endl;

  return num_updates;
}

template <class T, std::integral ID, class Distance = sum_of_squares_distance>
auto nn_descent(auto&& db, size_t k_nn, Distance distance = Distance()) {
  auto g = init_random_nn_graph<T, ID, Distance>(db, k_nn, distance);

  size_t num_updates{0};
  do {
    num_updates = nn_descent_step_all(g, db);
  } while (num_updates > (k_nn * num_vertices(g)) / 100);

  return g;
}

}  // namespace detail::graph

#endif  // TILEDB_NN_DESCENT_H
