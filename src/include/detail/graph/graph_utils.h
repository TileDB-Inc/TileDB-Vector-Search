/**
 * @file   graph_utils.h
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

#ifndef TDB_GRAPH_UTILS_H
#define TDB_GRAPH_UTILS_H

#include <fstream>
#include <string>

template <class Graph>
void dump_edgelist(const std::string& filename, const Graph& graph) {
  std::ofstream out(filename);

  out << "# source target\n";
  for (size_t i = 0; i < graph.num_vertices(); ++i) {
    for (auto&& [_, j] : out_edges(graph, i)) {
      out << i << " " << j << "\n";
    }
  }
  out.close();
}

template <class Distance = sum_of_squares_distance>
auto validate_graph(auto&& graph, auto&& X, Distance distance = Distance()) {
  using score_type = typename std::decay_t<decltype(graph)>::score_type;
  using id_type = typename std::decay_t<decltype(graph)>::id_type;

  using return_type = std::tuple<id_type, id_type, score_type, score_type>;
  auto return_vec = std::vector<return_type>{};
  for (size_t i = 0; i < graph.num_vertices(); ++i) {
    for (auto&& [s, j] : out_edges(graph, i)) {
      if (distance(X[i], X[j]) != s) {
        return_vec.emplace_back(i, j, s, distance(X[i], X[j]));
        std::cout << "i: " << i << " j: " << j << " s: " << s
                  << " d: " << distance(X[i], X[j]) << "\n";
      }
    }
  }
  return return_vec;
}

#endif  // TDB_GRAPH_UTILS_H
