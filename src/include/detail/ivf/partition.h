/**
 * @file   partition.h
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
 */

#ifndef TDB_IVF_PARTITION_H
#define TDB_IVF_PARTITION_H

#include <map>
#include <set>
#include <tuple>
#include <vector>
#include "detail/flat/qv.h"

namespace detail::ivf {

/**
 * @brief Get the top centroids for each query vector.
 * @tparam C
 * @tparam Q
 * @param centroids
 * @param query
 * @param nprobe
 * @param nthreads
 * @return The top nprobe centroids for each query vector.
 */
template <
    feature_vector_array C,
    query_vector_array Q,
    class Distance = sum_of_squares_distance>
auto ivf_top_centroids(
    const C& centroids,
    const Q& query,
    size_t nprobe,
    size_t nthreads,
    Distance distance = Distance{}) {
  return detail::flat::qv_query_heap_0(
      centroids, query, nprobe, nthreads, distance);
}

/**
 * In order to execute a query in distributed fashion, we perform a
 * preprocessing step on a single node to determine the total set of
 * partitions that need to be queried.  The resulting set of partitions
 * is, well, partitioned and a subset is sent to each compute node,
 * along with their associated centroids and queries.
 *
 * Each compute node performs a query on its subset of partitions and
 * returns its results to the master node.  The master node then merges
 * the results from each compute node and returns the final result.
 *
 */
template <
    class parts_type,
    feature_vector_array C,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto partition_ivf_flat_index(
    const C& centroids,
    const Q& query,
    size_t nprobe,
    size_t nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  if (::num_vectors(centroids) < nprobe) {
    throw std::invalid_argument(
        "nprobe (" + std::to_string(nprobe) +
        ") must be less than the number of centroids (" +
        std::to_string(::num_vectors(centroids)) + ")");
  }

  size_t num_queries = num_vectors(query);

  // Get the closest centroid for each query vector
  // There may be duplicates
  auto top_centroids = ivf_top_centroids(centroids, query, nprobe, nthreads);

  /*
   * `top_centroids` maps from rank X query index to the centroid *index*.
   *
   * To process centroids (partitions) in order, we need to map from `centroid`
   * to the set of queries having that centroid.
   *
   * We also need to know the "active" centroids, i.e., the ones having at
   * least one query.
   */
  auto centroid_query = std::multimap<parts_type, size_t>{};
  auto active_centroids = std::set<parts_type>{};
  for (size_t j = 0; j < num_queries; ++j) {
    for (size_t p = 0; p < nprobe; ++p) {
      auto tmp = top_centroids(p, j);
      // assert(tmp == 0);
      centroid_query.emplace(top_centroids(p, j), j);
      active_centroids.emplace(top_centroids(p, j));
    }
  }

  /*
   * From the active centroids we can compute the active partitions, i.e.,
   * the partitions that have at least one query.  Note that since the
   * active centroids were stored by index in a set, the active partitions
   * will be stored in sorted order.
   */
  auto active_partitions =
      std::vector<parts_type>(begin(active_centroids), end(active_centroids));

  /*
   * Get the query vectors associated with each partition.  We store the
   * index of the query vectors in the original query matrix.  We are
   * basically creating a partitionable version of the centroid_query map.
   */
  std::vector<std::vector<parts_type>> part_queries(size(active_partitions));

  for (size_t partno = 0; partno < size(active_partitions); ++partno) {
    auto active_part = active_partitions[partno];
    auto num_part_queries = centroid_query.count(active_part);
    part_queries[partno].reserve(num_part_queries);
    auto range = centroid_query.equal_range(active_part);
    for (auto i = range.first; i != range.second; ++i) {
      part_queries[partno].emplace_back(i->second);
    }
  }

  return std::make_tuple(std::move(active_partitions), std::move(part_queries));
}
}  // namespace detail::ivf

#endif  // TDB_IVF_PARTITION_H
