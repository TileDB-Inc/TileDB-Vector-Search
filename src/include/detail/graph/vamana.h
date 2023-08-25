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

#include <cstdint>
using size_t = std::size_t;

#include <queue>

#include "utils/fixed_min_heap.h"
#include "scoring.h"

/**
 * @brief
 * @tparam T
 * @tparam I
 * @param graph
 * @param source start node index
 * @param query query node index
 * @param k result size
 * @param L search list size, L >= k
 * @return result set ell containing k-approximate nearest neighbors to query and set vee
 * containing all the visited nodes
 *
 * Per the paper, the algorithm is as follows:
 * 1. Initialize the result list with the source node and visited list with empty
 * 2. While the result list \ visited list is not empty
 *    a. Find p* in the result list \ visited list with the smallest distance to the query
 *    b. update the result list with the out neighbors of p*
 *    c. Add p* to the visited list
 *    d. If size of the result list > L, trim the result list to keep L closest points to query
 * 3. Copy the result list to the output
 */
template <class I = size_t, class Distance = sum_of_squares_distance>
auto greedy_search(auto&& graph, auto&&db, I source, auto&& query, size_t k, size_t L, auto&& nbd, Distance&& distance = Distance{}) {
  using value_type = typename std::decay_t<decltype(graph)>::value_type;

  std::set<I> visited_vertices;
  auto visited = [&visited_vertices](auto&& v) {
    return visited_vertices.find(v) != visited_vertices.end();
  };

  auto result = k_min_heap<value_type, I> {k};
  auto q1 = k_min_heap<value_type, I> {L};

  result.insert(distance(db[source], query), source);
  q1.insert(distance(db[source], query), source);

  while(!q1.empty()) {
    // p_star is the nearest unvisited neighbor to the query
    auto [s_star, p_star] = q1.front();
    q1.pop();

    for (auto&& [_, p] : graph.out_edges(p_star)) {
      if (visited(p)) {
        continue;
      }
      auto score = distance(db[p], query);
      q1.insert(score, p);
      result.insert(score, p);
      visited_vertices.insert(p_star);
    }
  }

  get_top_k_from_heap(result, nbd, k);
}