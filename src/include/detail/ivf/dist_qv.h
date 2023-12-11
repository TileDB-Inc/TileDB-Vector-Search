/**
 * @file   ivf/dist_qv.h
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
 * Implementation of infrastructure for distributed queries, based on
 * qv_finite_ram.
 */

#ifndef TILEDB_IVF_DIST_QV_H
#define TILEDB_IVF_DIST_QV_H

#include <future>
#include <string>
#include <vector>

#include <tiledb/tiledb>

#include "detail/ivf/index.h"
#include "detail/ivf/partition.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "stats.h"
#include "utils/fixed_min_heap.h"

#include "detail/ivf/qv.h"

namespace detail::ivf {

/**
 * Function to be run on a distributed compute node in a TileDB task graph.
 * @param active_partitions -- A vector of which partitions in the part_uri
 * array to query against
 * @param query -- The full set of query vectors
 * @param active_queries -- A vector of which query vectors to use
 * @param indices -- The full set of indices
 * @param nthreads -- The number of threads to be executed on the compute node
 * @return A vector of min heaps, one for each query vector
 *
 * @todo Be more parsimonious about parameters and return values.
 * Should be able to use only the indices and the query vectors that are active
 * on this node, and return only the min heaps for the active query vectors
 */
template <class feature_type, class shuffled_ids_type>
auto dist_qv_finite_ram_part(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& dist_active_partitions,
    auto&& query,
    auto&& dist_active_queries,
    auto&& global_indices,
    const std::string& id_uri,
    size_t k_nn,
    uint64_t timestamp = 0,
    size_t nthreads = std::thread::hardware_concurrency()) {
  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }
  auto temporal_policy =
      (timestamp == 0) ? tiledb::TemporalPolicy() :
                         tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp);

  using score_type = float;
  // using parts_type = typename
  // std::remove_reference_t<decltype(dist_active_partitions)> ::value_type;
  using indices_type =
      typename std::remove_reference_t<decltype(global_indices)>::value_type;

  size_t num_queries = num_vectors(query);

  /*
   * We are given a subset of the partitions of the array, as given by
   * `dist_active_partitions`.  We create a partitioned matrix comprised of
   * those partitions.
   */
  auto partitioned_vectors = tdbColMajorPartitionedMatrix<
      feature_type,
      shuffled_ids_type,
      indices_type>(
      ctx,
      part_uri,
      global_indices,
      id_uri,
      dist_active_partitions,
      0,
      temporal_policy);

  // We are assuming that we are not doing out of core computation here.
  // (It is easy enough to change this if we need to.)
  partitioned_vectors.load();

  scoped_timer _i{tdb_func__ + " in RAM"};

  auto min_scores =
      std::vector<fixed_min_pair_heap<score_type, shuffled_ids_type>>(
          num_queries,
          fixed_min_pair_heap<score_type, shuffled_ids_type>(k_nn));

  assert(::num_partitions(partitioned_vectors) == size(dist_active_partitions));

  auto current_part_size = ::num_partitions(partitioned_vectors);
  size_t parts_per_thread = (current_part_size + nthreads - 1) / nthreads;

  std::vector<std::future<decltype(min_scores)>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto first_part = std::min<size_t>(n * parts_per_thread, current_part_size);
    auto last_part =
        std::min<size_t>((n + 1) * parts_per_thread, current_part_size);

    if (first_part != last_part) {
      futs.emplace_back(std::async(
          std::launch::async,
          [&query,
           &partitioned_vectors,
           &active_queries = dist_active_queries,
           k_nn,
           first_part,
           last_part]() {
            return apply_query(
                partitioned_vectors,
                std::optional<std::vector<int>>{},
                // std::optional{active_partitions},
                query,
                active_queries,
                k_nn,
                first_part,
                last_part);
          }));
    }
  }

  for (size_t n = 0; n < size(futs); ++n) {
    auto min_n = futs[n].get();

    for (size_t j = 0; j < num_queries; ++j) {
      for (auto&& [e, f] : min_n[j]) {
        min_scores[j].insert(e, f);
      }
    }
  }
  return min_scores;
}

#if 0
  auto min_scores =
      std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>>(
          nthreads,
          std::vector<fixed_min_pair_heap<score_type, id_type>>(
              num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  size_t parts_per_thread =
      (partitioned_vectors.num_col_parts() + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto first_part =
        std::min<size_t>(n * parts_per_thread, partitioned_vectors.num_col_parts());
    auto last_part = std::min<size_t>(
        (n + 1) * parts_per_thread, partitioned_vectors.num_col_parts());

    if (first_part != last_part) {
      futs.emplace_back(std::async(
          std::launch::async,
          [&, &active_queries = active_queries, n, first_part, last_part]() {
            /*
             * For each partition, process the queries that have that
             * partition as their top centroid.
             */
            for (size_t p = first_part; p < last_part; ++p) {
              auto partno = p + partitioned_vectors.col_part_offset();
              auto start = new_indices[partno] - partitioned_vectors.col_offset();
              auto stop = new_indices[partno + 1] - partitioned_vectors.col_offset();

              /*
               * Get the queries associated with this partition.
               */
              //
              for (auto j : active_queries[partno]) {
                auto q_vec = query[j];

                /*
                 * Apply the query to the partition.
                 */
                for (size_t kp = start; kp < stop; ++kp) {
                  auto score = L2(q_vec, partitioned_vectors[kp]);

                  // @todo any performance with apparent extra indirection?
                  min_scores[n][j].insert(score, partitioned_vectors.ids()[kp]);
                }
              }
            }
          }));
    }
  }

  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].get();
  }

  auto min_min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn));

  for (size_t j = 0; j < num_queries; ++j) {
    for (size_t n = 0; n < nthreads; ++n) {
      for (auto&& [e0, e1] : min_scores[n][j]) {
        min_min_scores[j].insert(e0, e1);
      }
    }
  }

  return min_min_scores;
#endif

template <class feature_type, class shuffled_ids_type>
auto dist_qv_finite_ram(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& query,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads,
    size_t num_nodes,
    uint64_t timestamp = 0) {
  scoped_timer _{tdb_func__ + " " + part_uri};

  // Check that the size of the indices vector is correct
  assert(size(indices) == centroids.num_cols() + 1);

  using score_type = float;
  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;
  using parts_type = indices_type;

  auto num_queries = num_vectors(query);

  /*
   * Select the relevant partitions from the array, along with the queries
   * that are in turn relevant to each partition.
   */
  auto&& [active_partitions, active_queries] =
      partition_ivf_flat_index<parts_type>(centroids, query, nprobe, nthreads);

  auto num_parts = size(active_partitions);

  auto min_scores =
      std::vector<fixed_min_pair_heap<score_type, shuffled_ids_type>>(
          num_queries,
          fixed_min_pair_heap<score_type, shuffled_ids_type>(k_nn));

  size_t parts_per_node = (num_parts + num_nodes - 1) / num_nodes;

  /*
   * "Distribute" the work to compute nodes.
   */
  for (size_t node = 0; node < num_nodes; ++node) {
    auto first_part = std::min<size_t>(node * parts_per_node, num_parts);
    auto last_part = std::min<size_t>((node + 1) * parts_per_node, num_parts);

    if (first_part != last_part) {
      auto dist_partitions = std::vector<parts_type>{
          begin(active_partitions) + first_part,
          begin(active_partitions) + last_part};
      auto dist_indices = std::vector<indices_type>{
          begin(indices) + first_part, begin(indices) + last_part + 1};
      auto dist_active_queries = std::vector<std::vector<indices_type>>{
          begin(active_queries) + first_part,
          begin(active_queries) + last_part};

      /*
       * Each compute node returns a min_heap of its own min_scores
       */

#if 1
      auto dist_min_scores =
          dist_qv_finite_ram_part<feature_type, shuffled_ids_type>(
              ctx,
              part_uri,
              dist_partitions,
              query,
              dist_active_queries,
              indices,
              id_uri,
              k_nn,
              timestamp,
              nthreads);
#else
      auto temporal_policy =
          (timestamp == 0) ?
              tiledb::TemporalPolicy() :
              tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp);
      auto partitioned_vectors = tdbColMajorPartitionedMatrix<
          feature_type,
          shuffled_ids_type,
          indices_type>(
          ctx, part_uri, indices, id_uri, dist_partitions, 0, temporal_policy);

      partitioned_vectors.load();
      auto&& [dist_min_scores, _] = query_finite_ram(
          partitioned_vectors, query, active_queries, k_nn, 0, nthreads);
#endif

      /*
       * Merge the min_scores from each compute node.
       */
      for (size_t j = 0; j < num_queries; ++j) {
        for (auto&& [e0, e1] : dist_min_scores[j]) {
          min_scores[j].insert(e0, e1);
        }
      }
    }
  }

  auto top_k = get_top_k_with_scores(min_scores, k_nn);

  return top_k;
}

}  // namespace detail::ivf

#endif  // TILEDB_IVF_DIST_QV_H
