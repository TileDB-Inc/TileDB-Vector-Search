/**
 * @file   qv.h
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
 * Implementation of queries for the "qv" orderings, i.e., for which the loop
 * over the queries is on the outer loop and the loop over the vectors is on the
 * inner loop.  Since the vectors in the inner loop are partitioned, we can
 * operate on them blockwise, which ameliorates some of the locality issues that
 * arise when doing this order with a flat index, say.
 *
 * There are two types of implementation here: infinite RAM and finite RAM.  The
 * infinite RAM case loads the entire partitioned database into memory, and then
 * searches in the partitions indicated by the given active partitions.
 * The infinite RAM case does not perform any out-of-core operations.
 * The finite RAM only loads the partitions into memory that are necessary
 * for the search, i.e., all partitions are active partitions. The user can
 * also specify an upper bound on the amount of RAM to be used for holding the
 * partitions to be searched (i.e., can specify amount of RAM to be used,
 * resulting in out of core operation.  The queries are ordered so
 * that the partitions can be loaded into memory in the order they are laid out
 * in the array.
 *
 * In general there is probably no reason to ever use the infinite RAM case
 * other than for benchmarking, as it requires machines with very large amounts
 * of RAM.
 */

#ifndef TILEDB_IVF_QV_H
#define TILEDB_IVF_QV_H

#include <future>
#include <map>
#include <string>

#include "algorithm.h"
#include "concepts.h"
#include "cpos.h"
#include "detail/ivf/partition.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "linalg.h"
#include "scoring.h"

#include "utils/print_types.h"

namespace detail::ivf {

/*******************************************************************************
 *
 * Functions for searching with infinite RAM
 *
 ******************************************************************************/

/**
 * @brief The (OG)^2 version of querying with qv loop ordering.
 * Applies a set of query vectors against a partitioned vector database.
 * Since this is an infinite ram algorithm, all of the partitions have been
 * loaded into memory. The partitions to apply the query to are inferred from
 * the top centroids.
 *
 * @tparam F Type of the feature vectors
 * @tparam C Type of the top centroids
 * @tparam Q Type of the query
 * @param top_centroids The centroids involved in the query
 * @param partitioned_vectors The partitioned database (all vectors)
 * @param query The query to be applied
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param nthreads How many threads to parallelize with
 * @return tuple of the top_k scores and the top_k indices
 *
 * @todo Can we active partitions rather than top_centroids?
 */
template <
    feature_vector_array C,
    partitioned_feature_vector_array F,
    query_vector_array Q,
    class Distance = sum_of_squares_distance>
auto qv_query_heap_infinite_ram(
    const C& top_centroids,
    const F& partitioned_vectors,
    const Q& query,
    size_t nprobe,
    size_t k_nn,
    size_t nthreads,
    Distance distance = Distance{}) {
  if (num_loads(partitioned_vectors) == 0) {
    load(partitioned_vectors);
  }
  scoped_timer _{tdb_func__};

  using id_type = typename F::id_type;
  auto indices = partitioned_vectors.indices();
  auto partitioned_ids = partitioned_vectors.ids();

  using score_type = float;

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_vectors(query), fixed_min_pair_heap<score_type, id_type>(k_nn));

  // Parallelize over the queries
  {
    auto par = stdx::execution::indexed_parallel_policy{nthreads};
    stdx::range_for_each(
        std::move(par),
        query,
        [&, nprobe](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
          for (size_t p = 0; p < nprobe; ++p) {
            auto aa = top_centroids(p, j);
            if (top_centroids(p, j) >= size(indices) - 1) {
              throw std::runtime_error(
                  "[qv_query_heap_infinite_ram] top_centroids(p, j) >= "
                  "size(indices) - 1");
            }

            size_t start = indices[top_centroids(p, j)];
            size_t stop = indices[top_centroids(p, j) + 1];

            for (size_t i = start; i < stop; ++i) {
              auto score = distance(q_vec /*q[j]*/, partitioned_vectors[i]);
              min_scores[j].insert(score, partitioned_ids[i]);
            }
          }
        });
  }

  return get_top_k_with_scores(min_scores, k_nn);
}

/**
 * @brief Query using the new qv (nuv) loop ordering.
 * Applies a set of query vectors against a partitioned vector database.
 * Since this is an infinite ram algorithm, all of the partitions have been
 * loaded into memory. The partitions to apply the query to are inferred from
 * the top centroids.
 *
 * @tparam F Type of the feature vectors
 * @tparam Q Type of the query
 * @param partitioned_vectors The partitioned database (all vectors)
 * @param active_partitions The partitions to which queries will be applied
 * @param query The query to be applied
 * @param active_queries The queries associated with each (active) partition
 * @param k_nn How many nearest neighbors to return
 * @param nthreads How many threads to parallelize with
 * @return tuple of the top_k scores and the top_k indices
 *
 */
template <
    partitioned_feature_vector_array F,
    query_vector_array Q,
    class Distance = sum_of_squares_distance>
auto nuv_query_heap_infinite_ram(
    const F& partitioned_vectors,
    auto&& active_partitions,
    const Q& query,
    auto&& active_queries,
    size_t k_nn,
    size_t nthreads,
    Distance distance = Distance{}) {
  if (num_loads(partitioned_vectors) == 0) {
    load(partitioned_vectors);
  }
  scoped_timer _{tdb_func__ + std::string{"_in_ram"}};

  using id_type = typename F::id_type;
  using score_type = float;

  auto partitioned_ids = partitioned_vectors.ids();
  auto indices = partitioned_vectors.indices();

  auto num_queries = num_vectors(query);

  auto min_scores =
      std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>>(
          nthreads,
          std::vector<fixed_min_pair_heap<score_type, id_type>>(
              num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  size_t parts_per_thread = (size(active_partitions) + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto first_part =
        std::min<size_t>(n * parts_per_thread, size(active_partitions));
    auto last_part =
        std::min<size_t>((n + 1) * parts_per_thread, size(active_partitions));

    if (first_part != last_part) {
      futs.emplace_back(std::async(
          std::launch::async,
          [&,
           &active_queries = active_queries,
           &active_partitions = active_partitions,
           n,
           first_part,
           last_part]() {
            /*
             * For each partition, process the associated queries
             */
            for (size_t p = first_part; p < last_part; ++p) {
              auto partno = active_partitions[p];
              auto start = indices[partno];
              auto stop = indices[partno + 1];

              /*
               * Get the queries associated with this partition.
               */
              for (auto j : active_queries[p]) {
                auto q_vec = query[j];

                // for (size_t k = start; k < stop; ++k) {
                //   auto kp = k - partitioned_vectors.col_offset();
                for (size_t kp = start; kp < stop; ++kp) {
                  auto score = distance(q_vec, partitioned_vectors[kp]);

                  // @todo any performance with apparent extra indirection?
                  // (Compiler should do the right thing, but...)
                  min_scores[n][j].insert(score, partitioned_ids[kp]);
                }
              }
            }
          }));
    }
  }
  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].get();
  }

  consolidate_scores(min_scores);
  return get_top_k_with_scores(min_scores, k_nn);
}

/**
 * @brief Query using the new qv (nuv) loop ordering, with 2 by 2 blocking.  The
 * blocking is applied in order to increase locality.
 *
 * Applies a set of query vectors against a partitioned vector database.
 * Since this is an infinite ram algorithm, all of the partitions have been
 * loaded into memory. The partitions to apply the query to are inferred from
 * the top centroids.
 *
 * @tparam F Type of the feature vectors
 * @tparam Q Type of the query
 * @param partitioned_vectors The partitioned database (all vectors)
 * @param active_partitions The partitions to which queries will be applied
 * @param query The query to be applied
 * @param active_queries The queries associated with each (active) partition
 * @param k_nn How many nearest neighbors to return
 * @param nthreads How many threads to parallelize with
 * @return tuple of the top_k scores and the top_k indices
 *
 */
template <
    feature_vector_array F,
    query_vector_array Q,
    class Distance = sum_of_squares_distance>
auto nuv_query_heap_infinite_ram_reg_blocked(
    const F& partitioned_vectors,
    auto&& active_partitions,
    const Q& query,
    auto&& active_queries,
    size_t k_nn,
    size_t nthreads,
    Distance distance = Distance{}) {
  if (num_loads(partitioned_vectors) == 0) {
    load(partitioned_vectors);
  }
  scoped_timer _{tdb_func__ + std::string{"_in_ram"}};

  using id_type = typename F::id_type;
  using score_type = float;

  auto partitioned_ids = partitioned_vectors.ids();
  auto indices = partitioned_vectors.indices();

  auto num_queries = num_vectors(query);

  auto min_scores =
      std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>>(
          nthreads,
          std::vector<fixed_min_pair_heap<score_type, id_type>>(
              num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  size_t parts_per_thread = (size(active_partitions) + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto first_part =
        std::min<size_t>(n * parts_per_thread, size(active_partitions));
    auto last_part =
        std::min<size_t>((n + 1) * parts_per_thread, size(active_partitions));

    if (first_part != last_part) {
      futs.emplace_back(std::async(
          std::launch::async,
          [&,
           &active_queries = active_queries,
           &active_partitions = active_partitions,
           n,
           first_part,
           last_part]() {
            /*
             * For each partition, process the associated queries
             */
            auto& mscores = min_scores[n];
            for (size_t partno = first_part; partno < last_part; ++partno) {
              auto quartno = active_partitions[partno];
              auto start = indices[quartno];
              auto stop = indices[quartno + 1];
              auto kstep = stop - start;
              auto kstop = start + 2 * (kstep / 2);

              auto len = 2 * (size(active_queries[partno]) / 2);
              auto end = active_queries[partno].begin() + len;
              for (auto j = active_queries[partno].begin(); j != end; j += 2) {
                auto j0 = j[0];
                auto j1 = j[1];
                auto q_vec_0 = query[j0];
                auto q_vec_1 = query[j1];

                for (size_t kp = start; kp < kstop; kp += 2) {
                  auto score_00 =
                      distance(q_vec_0, partitioned_vectors[kp + 0]);
                  auto score_01 =
                      distance(q_vec_0, partitioned_vectors[kp + 1]);
                  auto score_10 =
                      distance(q_vec_1, partitioned_vectors[kp + 0]);
                  auto score_11 =
                      distance(q_vec_1, partitioned_vectors[kp + 1]);

                  min_scores[n][j0].insert(score_00, partitioned_ids[kp + 0]);
                  min_scores[n][j0].insert(score_01, partitioned_ids[kp + 1]);
                  min_scores[n][j1].insert(score_10, partitioned_ids[kp + 0]);
                  min_scores[n][j1].insert(score_11, partitioned_ids[kp + 1]);
                }

                /*
                 * Cleanup the last iteration(s) of k
                 */
                for (size_t kp = kstop; kp < stop; ++kp) {
                  auto score_00 =
                      distance(q_vec_0, partitioned_vectors[kp + 0]);
                  auto score_10 =
                      distance(q_vec_1, partitioned_vectors[kp + 0]);
                  min_scores[n][j0].insert(score_00, partitioned_ids[kp + 0]);
                  min_scores[n][j1].insert(score_10, partitioned_ids[kp + 0]);
                }
              }

              /*
               * Cleanup the last iteration(s) of j
               */
              for (auto j = end; j < active_queries[partno].end(); ++j) {
                auto j0 = j[0];
                auto q_vec_0 = query[j0];

                for (size_t kp = start; kp < kstop; kp += 2) {
                  auto score_00 =
                      distance(q_vec_0, partitioned_vectors[kp + 0]);
                  auto score_01 =
                      distance(q_vec_0, partitioned_vectors[kp + 1]);

                  min_scores[n][j0].insert(score_00, partitioned_ids[kp + 0]);
                  min_scores[n][j0].insert(score_01, partitioned_ids[kp + 1]);
                }
                for (size_t kp = kstop; kp < stop; ++kp) {
                  auto score_00 =
                      distance(q_vec_0, partitioned_vectors[kp + 0]);
                  min_scores[n][j0].insert(score_00, partitioned_ids[kp + 0]);
                }
              }
            }
          }));
    }
  }
  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].get();
  }

  consolidate_scores(min_scores);
  return get_top_k_with_scores(min_scores, k_nn);
}

/*******************************************************************************
 *
 * Functions for searching with finite RAM
 *
 ******************************************************************************/

// ----------------------------------------------------------------------------
// Functions for searching with finite RAM, new qv (nuv) ordering
// ----------------------------------------------------------------------------

/*
 * Queries a set of query vectors against an indexed vector database, using
 * the "nuv" ordering.
 *
 *  Queries a set of query vectors against an indexed vector database. The
 * arrays part_uri, centroids, indices, and id_uri comprise the index.  The
 * partitioned database is stored in part_uri, the centroids are stored in
 * centroids, the indices demarcating partitions is stored in indices, and the
 * labels for the vectors in the original database are stored in id_uri.
 * The query is stored in q.
 *
 * "Finite RAM" means the only the partitions necessary for the search will
 * be loaded into memory.  Moreover, the `upper_bound` parameter controls
 * how many vectors will be loaded into memory at any one time.  If all of
 * the partitions cannot be loaded into memory at once, then batches of them
 * will be loaded, and the search will be performed on the batches.  (That is,
 * the search will be performed "out of core".)  An upper_bound of 0 will load
 * only the partitions necessary to answer the query, but it will load all of
 * them.
 *
 * The function loads partitions in the order they are stored in the array.
 *
 * @tparam F Type of the feature vectors
 * @tparam Q Type of the query
 * @param partitioned_vectors The partitioned database (all vectors)
 * @param active_partitions The partitions to which queries will be applied
 * @param query The query to be applied
 * @param active_queries The queries associated with each (active) partition
 * @param k_nn How many nearest neighbors to return
 * @param nthreads How many threads to parallelize with
 * @return tuple of the top_k scores and the top_k indices
 *
 */
template <
    feature_vector_array F,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto nuv_query_heap_finite_ram(
    F& partitioned_vectors,
    const Q& query,
    auto&& active_queries,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  using id_type = typename F::id_type;
  using score_type = float;

  auto num_queries = num_vectors(query);

  auto min_scores =
      std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>>(
          nthreads,
          std::vector<fixed_min_pair_heap<score_type, id_type>>(
              num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  log_timer _i{tdb_func__ + " in RAM"};

  size_t part_offset = 0;
  while (partitioned_vectors.load()) {
    _i.start();

    auto indices = partitioned_vectors.indices();
    auto partitioned_ids = partitioned_vectors.ids();

    auto current_part_size = ::num_partitions(partitioned_vectors);
    size_t parts_per_thread = (current_part_size + nthreads - 1) / nthreads;

    std::vector<std::future<void>> futs;
    futs.reserve(nthreads);

    for (size_t n = 0; n < nthreads; ++n) {
      auto first_part =
          std::min<size_t>(n * parts_per_thread, current_part_size);
      auto last_part =
          std::min<size_t>((n + 1) * parts_per_thread, current_part_size);

      if (first_part != last_part) {
        futs.emplace_back(std::async(
            std::launch::async,
            [&,
             &active_queries = active_queries,
             n,
             first_part,
             last_part,
             part_offset]() {
              /*
               * For each partition, process the associated queries
               */
              for (size_t p = first_part; p < last_part; ++p) {
                // @todo this may not be correct -- resident_part_offset is
                // part of tdbPartitionedMatrix, not PartitionedMatrix
                auto partno =
                    p +
                    part_offset;  // resident_part_offset(partitioned_vectors);
                auto start = indices[p];     //[partno];
                auto stop = indices[p + 1];  //[partno+1];

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
                    auto score = distance(q_vec, partitioned_vectors[kp]);

                    // @todo any performance with apparent extra indirection?
                    min_scores[n][j].insert(
                        score, partitioned_vectors.ids()[kp]);
                  }
                }
              }
            }));
      }
    }

    for (size_t n = 0; n < size(futs); ++n) {
      futs[n].get();
    }
    _i.stop();
    part_offset += current_part_size;
  }

  consolidate_scores(min_scores);
  return get_top_k_with_scores(min_scores, k_nn);
}

/**
 * Queries a set of query vectors against an indexed vector database, using
 * "nuv" ordering and 2 by 2 blocking.  The blocking is used in order to
 * increase locality.
 *
 * "Finite RAM" means the only the partitions necessary for the search will
 * be loaded into memory.  Moreover, the `upper_bound` parameter controls
 * how many vectors will be loaded into memory at any one time.  If all of
 * the partitions cannot be loaded into memory at once, then batches of them
 * will be loaded, and the search will be performed on the batches.  (That is,
 * the search will be performed "out of core".)  An upper_bound of 0 will load
 * only the partitions necessary to answer the query, but it will load all of
 * them.
 *
 * The function loads partitions in the order they are stored in the array.
 *
 * @tparam F Type of the feature vectors
 * @tparam Q Type of the query
 * @param partitioned_vectors The partitioned database (all vectors)
 * @param active_partitions The partitions to which queries will be applied
 * @param query The query to be applied
 * @param active_queries The queries associated with each (active) partition
 * @param k_nn How many nearest neighbors to return
 * @param nthreads How many threads to parallelize with
 * @return tuple of the top_k scores and the top_k indices
 *
 */
template <
    feature_vector_array F,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto nuv_query_heap_finite_ram_reg_blocked(
    F& partitioned_vectors,  // not const because of load()
    const Q& query,
    auto&& active_queries,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  using id_type = typename F::id_type;
  using score_type = float;

  auto num_queries = num_vectors(query);

  std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>> min_scores(
      nthreads,
      std::vector<fixed_min_pair_heap<score_type, id_type>>(
          num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  log_timer _i{tdb_func__ + " in RAM"};

  size_t part_offset = 0;
  while (partitioned_vectors.load()) {
    _i.start();

    auto indices = partitioned_vectors.indices();
    auto partitioned_ids = partitioned_vectors.ids();

    auto current_part_size = ::num_partitions(partitioned_vectors);
    size_t parts_per_thread = (current_part_size + nthreads - 1) / nthreads;

    std::vector<std::future<void>> futs;
    futs.reserve(nthreads);

    for (size_t n = 0; n < nthreads; ++n) {
      auto first_part =
          std::min<size_t>(n * parts_per_thread, current_part_size);
      auto last_part =
          std::min<size_t>((n + 1) * parts_per_thread, current_part_size);

      if (first_part != last_part) {
        futs.emplace_back(std::async(
            std::launch::async,
            [&min_scores,
             &query,
             &partitioned_vectors,
             &indices,
             &active_queries = active_queries,
             &distance,
             n,
             first_part,
             last_part,
             part_offset]() {
              /*
               * For each partition, process the associated queries that have
               * that
               *
               * Assumption: Each partition is an active partition
               */
              for (size_t p = first_part; p < last_part; ++p) {
                // @todo this may not be correct -- resident_part_offset is
                // part of tdbPartitionedMatrix, not PartitionedMatrix
                auto partno =
                    p +
                    part_offset;  // resident_part_offset(partitioned_vectors);

                if (p + 1 >= size(indices)) {
                  throw std::runtime_error(
                      "[nuv_query_heap_finite_ram_reg_blocked] p + 1 >= "
                      "size(indices)");
                }

                auto start = indices[p];
                auto stop = indices[p + 1];

                auto kstep = stop - start;
                auto kstop = start + 2 * (kstep / 2);

                auto len = 2 * (size(active_queries[partno]) / 2);
                auto end = active_queries[partno].begin() + len;
                for (auto j = active_queries[partno].begin(); j != end;
                     j += 2) {
                  auto j0 = j[0];
                  auto j1 = j[1];
                  auto q_vec_0 = query[j0];
                  auto q_vec_1 = query[j1];

                  for (size_t kp = start; kp < kstop; kp += 2) {
                    auto score_00 =
                        distance(q_vec_0, partitioned_vectors[kp + 0]);
                    auto score_01 =
                        distance(q_vec_0, partitioned_vectors[kp + 1]);
                    auto score_10 =
                        distance(q_vec_1, partitioned_vectors[kp + 0]);
                    auto score_11 =
                        distance(q_vec_1, partitioned_vectors[kp + 1]);

                    min_scores[n][j0].insert(
                        score_00, partitioned_vectors.ids()[kp + 0]);
                    min_scores[n][j0].insert(
                        score_01, partitioned_vectors.ids()[kp + 1]);
                    min_scores[n][j1].insert(
                        score_10, partitioned_vectors.ids()[kp + 0]);
                    min_scores[n][j1].insert(
                        score_11, partitioned_vectors.ids()[kp + 1]);
                  }

                  /*
                   * Cleanup the last iteration(s) of k
                   */
                  for (size_t kp = kstop; kp < stop; ++kp) {
                    auto score_00 =
                        distance(q_vec_0, partitioned_vectors[kp + 0]);
                    auto score_10 =
                        distance(q_vec_1, partitioned_vectors[kp + 0]);
                    min_scores[n][j0].insert(
                        score_00, partitioned_vectors.ids()[kp + 0]);
                    min_scores[n][j1].insert(
                        score_10, partitioned_vectors.ids()[kp + 0]);
                  }
                }

                /*
                 * Cleanup the last iteration(s) of j
                 */
                for (auto j = end; j < active_queries[partno].end(); ++j) {
                  auto j0 = j[0];
                  auto q_vec_0 = query[j0];

                  for (size_t kp = start; kp < kstop; kp += 2) {
                    auto score_00 =
                        distance(q_vec_0, partitioned_vectors[kp + 0]);
                    auto score_01 =
                        distance(q_vec_0, partitioned_vectors[kp + 1]);

                    min_scores[n][j0].insert(
                        score_00, partitioned_vectors.ids()[kp + 0]);
                    min_scores[n][j0].insert(
                        score_01, partitioned_vectors.ids()[kp + 1]);
                  }
                  for (size_t kp = kstop; kp < stop; ++kp) {
                    auto score_00 =
                        distance(q_vec_0, partitioned_vectors[kp + 0]);
                    min_scores[n][j0].insert(
                        score_00, partitioned_vectors.ids()[kp + 0]);
                  }
                }
              }
            }));
      }
    }

    for (size_t n = 0; n < size(futs); ++n) {
      futs[n].get();
    }
    part_offset += current_part_size;
    _i.stop();
  }

  consolidate_scores(min_scores);
  return get_top_k_with_scores(min_scores, k_nn);
}

/**
 * @todo Modify for new ivf_index api
 *
 * @brief Formerly OG Implementation of finite RAM qv query, but now
 * dispatches to nuv_query_heap_finite_ram.
 *
 * "Finite RAM" means the only the partitions necessary for the search will
 * be loaded into memory.  Moreover, the `upper_bound` parameter controls
 * how many vectors will be loaded into memory at any one time.  If all of
 * the partitions can be loaded into memory at once, then batches of them
 * will be loaded, and the search will be performed on the batches.  (That is
 * the search will be performed "out of core".)
 *
 * The function loads partitions in the order they are stored in the array.
 *
 * Note that an upper_bound of 0 means that all relevant partitions will be
 * loaded. This differs from infinite RAM, which loads the entire partitioned
 * database.  An upper_bound of 0 will load only the partitions necessary
 * to answer the query, but it will load all of those.
 *
 * @param part_uri The URI of the partitioned database
 * @param centroids Centroids of the vectors in the original database (and
 * the partitioned database).  The ith centroid is the centroid of the ith
 * partition.
 * @param q The query to be searched
 * @param indices The demarcations of partitions
 * @param id_uri URI for the labels for the vectors in the original database
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param upper_bound Limit of how many vectors to load into memory at one time
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @return The indices of the top_k neighbors for each query vector
 */
template <
    class feature_type,
    class id_type,
    class Distance = sum_of_squares_distance>
auto qv_query_heap_finite_ram(
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
    uint64_t timestamp = 0,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};
  auto temporal_policy = (timestamp == 0) ?
                             TemporalPolicy() :
                             TemporalPolicy(TimeTravel, timestamp);

  // using score_type = float;
  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;
  using parts_type = indices_type;

  size_t num_queries = num_vectors(query);

  auto&& [active_partitions, active_queries] =
      detail::ivf::partition_ivf_flat_index<parts_type>(
          centroids, query, nprobe, nthreads);

  auto partitioned_vectors =
      tdbColMajorPartitionedMatrix<feature_type, id_type, indices_type>(
          ctx,
          part_uri,
          indices,
          id_uri,
          active_partitions,
          upper_bound,
          temporal_policy);

  return nuv_query_heap_finite_ram(
      partitioned_vectors,
      query,
      active_queries,
      k_nn,
      upper_bound,
      nthreads,
      distance);
}

/*******************************************************************************
 *
 * GM functions for finite and infinite queries
 *
 ******************************************************************************/

/**
 * Queries a set of query vectors against an indexed vector database, using
 * "nuv" ordering and 2 by 2 blocking.  The blocking is used in order to
 * increase locality.
 *
 * This function is the "definitive" finite ram query.
 *
 * "Finite RAM" means the only the partitions necessary for the search will
 * be loaded into memory.  Moreover, the `upper_bound` parameter controls
 * how many vectors will be loaded into memory at any one time.  If all of
 * the partitions cannot be loaded into memory at once, then batches of them
 * will be loaded, and the search will be performed on the batches.  (That is,
 * the search will be performed "out of core".)  An upper_bound of 0 will load
 * only the partitions necessary to answer the query, but it will load all of
 * them.
 *
 * The function loads partitions in the order they are stored in the array.
 *
 * @tparam F Type of the feature vectors
 * @tparam Q Type of the query
 * @param partitioned_vectors The partitioned database (all vectors)
 * @param active_partitions The partitions to which queries will be applied
 * @param query The query to be applied
 * @param active_queries The queries associated with each (active) partition
 * @param k_nn How many nearest neighbors to return
 * @param nthreads How many threads to parallelize with
 * @return tuple of the top_k scores and the top_k indices
 *
 */
template <
    partitioned_feature_vector_array P,
    query_vector_array Q,
    class A,
    class Distance = sum_of_squares_distance>
auto apply_query(
    const P& partitioned_vectors,
    const std::optional<A>&
        active_partitions,  // Needed for infinite case, not finite
    const Q& query,
    auto&& active_queries,
    size_t k_nn,
    size_t first_active_part,
    size_t last_active_part,
    size_t part_offset = 0,
    Distance distance = Distance{}) {
  using id_type = typename P::id_type;
  using score_type = float;

  auto partitioned_ids = partitioned_vectors.ids();  // These are global
  auto indices = partitioned_vectors.indices();      // These are local to p_v

  auto num_queries = num_vectors(query);

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn));

  // Iterate through the active partitions -- for a finite query, this is
  // all of the partitions.  For an infinite query, this is a subset, given
  // by `active_partitions`.
  for (size_t p = first_active_part; p < last_active_part; ++p) {
    // Note that in the infinite case, the active_partitions are a subset
    // of all the partitions.  In the finite case, all partitions are active.
    // auto quartno = active_partitions[partno];
    auto partno = p + part_offset;
    auto active_partno = p;
    if (active_partitions) {
      active_partno = (*active_partitions)[p];
    }

    // active_partitions is only for infinite case -- no offset
    // for finite case, all partitions are active -- but indices are local
    // finite: active_queries use p + part_offset

    // indices is local to partitioned_vectors
    auto start = indices[active_partno];
    auto stop = indices[active_partno + 1];

    auto kstep = stop - start;
    auto kstop = start + 2 * (kstep / 2);

    auto len = 2 * (size(active_queries[partno]) / 2);
    auto end = active_queries[partno].begin() + len;

    for (auto j = active_queries[partno].begin(); j < end; j += 2) {
      auto j0 = j[0];
      auto j1 = j[1];
      auto q_vec_0 = query[j0];
      auto q_vec_1 = query[j1];

      for (size_t kp = start; kp < kstop; kp += 2) {
        auto score_00 = distance(q_vec_0, partitioned_vectors[kp + 0]);
        auto score_01 = distance(q_vec_0, partitioned_vectors[kp + 1]);
        auto score_10 = distance(q_vec_1, partitioned_vectors[kp + 0]);
        auto score_11 = distance(q_vec_1, partitioned_vectors[kp + 1]);

        min_scores[j0].insert(score_00, partitioned_ids[kp + 0]);
        min_scores[j0].insert(score_01, partitioned_ids[kp + 1]);
        min_scores[j1].insert(score_10, partitioned_ids[kp + 0]);
        min_scores[j1].insert(score_11, partitioned_ids[kp + 1]);
      }

      /*
       * Cleanup the last iteration(s) of k
       */
      for (size_t kp = kstop; kp < stop; ++kp) {
        auto score_00 = distance(q_vec_0, partitioned_vectors[kp + 0]);
        auto score_10 = distance(q_vec_1, partitioned_vectors[kp + 0]);
        min_scores[j0].insert(score_00, partitioned_ids[kp + 0]);
        min_scores[j1].insert(score_10, partitioned_ids[kp + 0]);
      }
    }

    /*
     * Cleanup the last iteration(s) of j
     */
    for (auto j = end; j < active_queries[partno].end(); ++j) {
      auto j0 = j[0];
      auto q_vec_0 = query[j0];

      for (size_t kp = start; kp < kstop; kp += 2) {
        auto score_00 = distance(q_vec_0, partitioned_vectors[kp + 0]);
        auto score_01 = distance(q_vec_0, partitioned_vectors[kp + 1]);

        min_scores[j0].insert(score_00, partitioned_ids[kp + 0]);
        min_scores[j0].insert(score_01, partitioned_ids[kp + 1]);
      }
      for (size_t kp = kstop; kp < stop; ++kp) {
        auto score_00 = distance(q_vec_0, partitioned_vectors[kp + 0]);
        min_scores[j0].insert(score_00, partitioned_ids[kp + 0]);
      }
    }
  }

  return min_scores;
}

/**
 * @brief Reference implementation of finite RAM query using qv ordering.
 * It incorporates all of the empircally determined optimizations from the
 * other implementations in this file.
 *
 * Queries a set of query vectors against an indexed vector database. The
 * arrays part_uri, centroids, indices, and id_uri comprise the index.  The
 * partitioned database is stored in part_uri, the centroids are stored in
 * centroids, the indices demarcating partitions is stored in indices, and the
 * labels for the vectors in the original database are stored in id_uri.
 * The query is stored in q.
 *
 * "Finite RAM" means the only the partitions necessary for the search will
 * be loaded into memory.  Moreover, the `upper_bound` parameter controls
 * how many vectors will be loaded into memory at any one time.  If all of
 * the partitions can be loaded into memory at once, then batches of them
 * will be loaded, and the search will be performed on the batches.  (That is
 * the search will be performed "out of core".)
 *
 * The function loads partitions in the order they are stored in the array.
 *
 * Note that an upper_bound of 0 means that all relevant partitions will be
 * loaded. This differs from infinite RAM, which loads the entire partitioned
 * database.  An upper_bound of 0 will load only the partitions necessary
 * to answer the query, but it will load all of those.
 *
 * @param part_uri The URI of the partitioned database
 * @param centroids Centroids of the vectors in the original database (and
 * the partitioned database).  The ith centroid is the centroid of the ith
 * partition.
 * @param q The query to be searched
 * @param indices The demarcations of partitions
 * @param id_uri URI for the labels for the vectors in the original database
 * @param k_nn How many nearest neighbors to return
 * @param upper_bound Limit of how many vectors to load into memory at one time
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @param min_parts_per_thread Unused (WIP for threading heuristics)
 * @return The indices of the top_k neighbors for each query vector
 */
template <
    feature_vector_array F,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto query_finite_ram(
    F& partitioned_vectors,  // not const because of load()
    const Q& query,
    auto&& active_queries,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  using id_type = typename F::id_type;
  using score_type = float;

  auto num_queries = num_vectors(query);

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn));

  log_timer _i{tdb_func__ + " in RAM"};

  size_t part_offset = 0;
  while (partitioned_vectors.load()) {
    _i.start();

    auto indices = partitioned_vectors.indices();
    auto partitioned_ids = partitioned_vectors.ids();

    auto current_part_size = ::num_partitions(partitioned_vectors);
    size_t parts_per_thread = (current_part_size + nthreads - 1) / nthreads;

    std::vector<std::future<decltype(min_scores)>> futs;
    futs.reserve(nthreads);

    for (size_t n = 0; n < nthreads; ++n) {
      auto first_part =
          std::min<size_t>(n * parts_per_thread, current_part_size);
      auto last_part =
          std::min<size_t>((n + 1) * parts_per_thread, current_part_size);

      if (first_part != last_part) {
        futs.emplace_back(std::async(
            std::launch::async,
            [&query,
             &partitioned_vectors,
             &active_queries = active_queries,
             &distance,
             k_nn,
             first_part,
             last_part,
             part_offset]() {
              return apply_query(
                  partitioned_vectors,
                  std::optional<std::vector<int>>{},
                  query,
                  active_queries,
                  k_nn,
                  first_part,
                  last_part,
                  part_offset,
                  distance);
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

    part_offset += current_part_size;
    _i.stop();
  }

  return get_top_k_with_scores(min_scores, k_nn);
}

/**
 * @brief Query using the new qv (nuv) loop ordering.
 * Applies a set of query vectors against a partitioned vector database.
 * Since this is an infinite ram algorithm, all of the partitions have been
 * loaded into memory. The partitions to apply the query to are inferred from
 * the top centroids.
 *
 * This function is the definitive infinite ram query.
 *
 * @tparam F Type of the feature vectors
 * @tparam Q Type of the query
 * @param partitioned_vectors The partitioned database (all vectors)
 * @param active_partitions The partitions to which queries will be applied
 * @param query The query to be applied
 * @param active_queries The queries associated with each (active) partition
 * @param k_nn How many nearest neighbors to return
 * @param nthreads How many threads to parallelize with
 * @return tuple of the top_k scores and the top_k indices
 *
 */
template <
    feature_vector_array F,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto query_infinite_ram(
    const F& partitioned_vectors,
    auto&& active_partitions,
    const Q& query,
    auto&& active_queries,
    size_t k_nn,
    size_t nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__ + std::string{"_in_ram"}};

  using id_type = typename F::id_type;
  using score_type = float;

  auto partitioned_ids = partitioned_vectors.ids();
  auto indices = partitioned_vectors.indices();

  auto num_queries = num_vectors(query);

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn));

  size_t parts_per_thread = (size(active_partitions) + nthreads - 1) / nthreads;

  std::vector<std::future<decltype(min_scores)>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto first_part =
        std::min<size_t>(n * parts_per_thread, size(active_partitions));
    auto last_part =
        std::min<size_t>((n + 1) * parts_per_thread, size(active_partitions));

    if (first_part != last_part) {
      futs.emplace_back(std::async(
          std::launch::async,
          [&query,
           &partitioned_vectors,
           &active_queries = active_queries,
           &active_partitions = active_partitions,
           &distance,
           k_nn,
           first_part,
           last_part]() {
            return apply_query(
                partitioned_vectors,
                std::optional{active_partitions},
                query,
                active_queries,
                k_nn,
                first_part,
                last_part,
                0,
                distance);
          }));
    }
  }

  // @todo We should do this without putting all queries on every node
  for (size_t n = 0; n < size(futs); ++n) {
    auto min_n = futs[n].get();

    for (size_t j = 0; j < num_queries; ++j) {
      for (auto&& e : min_n[j]) {
        min_scores[j].insert(std::get<0>(e), std::get<1>(e));
      }
    }
  }

  return get_top_k_with_scores(min_scores, k_nn);
}

}  // namespace detail::ivf

#endif  // TILEDB_IVF_QV_H
