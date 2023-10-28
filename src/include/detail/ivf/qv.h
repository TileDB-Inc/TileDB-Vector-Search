/**
 * @file   ivf/qv.h
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
 * searches in the partitions as indicated by the nearest centroids to the
 * queries.  The infinite RAM case does not perform any out-of-core operations.
 * The finite RAM case only loads the partitions into memory that are necessary
 * for the search. The user can specify an upper bound on the amount of RAM to
 * be used for holding the queries being searched.  The searches are ordered so
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
#include "flat_query.h"
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
 * @brief The OG version of querying with qv loop ordering.
 * Queries a set of query vectors against an indexed vector database. The
 * arrays part_uri, centroids, indices, and id_uri comprise the index.  The
 * partitioned database is stored in part_uri, the centroids are stored in
 * centroids, the indices demarcating partitions is stored in indices, and the
 * labels for the vectors in the original database are stored in id_uri.
 * The query is stored in q.
 *
 * * "Infinite RAM" means the entire index is loaded into memory before any
 * queries are applied, regardless of which partitions are to be queried.
 *
 * @param partitioned_vectors Partitioned database
 * @param centroids Centroids of the vectors in the original database (and
 * the partitioned database).  The ith centroid is the centroid of the ith
 * partition.
 * @param q The query to be searched
 * @param indices The demarcations of partitions
 * @param partitioned_ids Labels for the vectors in the original database
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @return The indices of the top_k neighbors for each query vector
 */
// @todo We should still order the queries so partitions are searched in order
template <
    partitioned_feature_vector_array F,
    feature_vector_array C,
    query_vector_array Q>
auto qv_query_heap_infinite_ram(
    const F& partitioned_vectors,
    const C& centroids,
    const Q& q,
    size_t nprobe,
    size_t k_nn,
    size_t nthreads) {
  if (num_loads(partitioned_vectors) == 0) {
    load(partitioned_vectors);
  }
  scoped_timer _{"Total time " + tdb_func__};

  using id_type = typename F::id_type;
  auto indices = partitioned_vectors.indices();
  auto partitioned_ids = partitioned_vectors.ids();

  using score_type = float;

  // @todo is this the best (fastest) algorithm to use?  (it takes miniscule
  // time at rate)
  auto top_centroids =
      detail::flat::qv_query_heap_0(centroids, q, nprobe, nthreads);

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_vectors(q), fixed_min_pair_heap<score_type, id_type>(k_nn));

  // Parallelizing over q is not going to be the most efficient
  {
    scoped_timer __{tdb_func__ + std::string{"_in_ram"}};
    auto par = stdx::execution::indexed_parallel_policy{nthreads};
    stdx::range_for_each(
        std::move(par),
        q,
        [&, nprobe](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
          for (size_t p = 0; p < nprobe; ++p) {
            size_t start = indices[top_centroids(p, j)];
            size_t stop = indices[top_centroids(p, j) + 1];

            for (size_t i = start; i < stop; ++i) {
              auto score = L2(q_vec /*q[j]*/, partitioned_vectors[i]);
              min_scores[j].insert(score, partitioned_ids[i]);
            }
          }
        });
  }

  auto top_k = get_top_k_with_scores(min_scores, k_nn);
  return top_k;
}

/**
 * @brief Implementation with the new qv (nuv) loop ordering.
 * Queries a set of query vectors against an indexed vector database. The
 * arrays part_uri, centroids, indices, and id_uri comprise the index.  The
 * partitioned database is stored in part_uri, the centroids are stored in
 * centroids, the indices demarcating partitions is stored in indices, and the
 * labels for the vectors in the original database are stored in id_uri.
 * The query is stored in q.
 *
 * "Infinite RAM" means the entire index is loaded into memory before any
 * queries are applied, regardless of which partitions are to be queried.
 *
 * @param partitioned_vectors Partitioned database
 * @param centroids Centroids of the vectors in the original database (and
 * the partitioned database).  The ith centroid is the centroid of the ith
 * partition.
 * @param q The query to be searched
 * @param indices The demarcations of partitions
 * @param partitioned_ids Labels for the vectors in the original database
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @return The indices of the top_k neighbors for each query vector
 */
template <
    contiguous_partitioned_feature_vector_array F,
    feature_vector_array C,
    query_vector_array Q>
auto nuv_query_heap_infinite_ram(
    const F& partitioned_vectors,
    const C& centroids,
    const Q& query,
    size_t nprobe,
    size_t k_nn,
    size_t nthreads) {
  if (num_loads(partitioned_vectors) == 0) {
    load(partitioned_vectors);
  }
  scoped_timer _{tdb_func__ + std::string{"_in_ram"}};

  using id_type = typename F::id_type;
  auto indices = partitioned_vectors.indices();
  auto partitioned_ids = partitioned_vectors.ids();

  using score_type = float;

  auto num_queries = num_vectors(query);

  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

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
             * For each partition, process the queries that have that
             * partition as their top centroid.
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
                  auto score = L2(q_vec, partitioned_vectors[kp]);

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
  auto top_k = get_top_k_with_scores(min_scores, k_nn);

  return top_k;
}

/**
 * @brief Implementation with the new qv (nuv) loop ordering.  In this function
 * we apply a tiling to the inner loop (similar to the approach used in
 * matrix-matrix product).
 *
 * Queries a set of query vectors against an indexed vector database. The
 * arrays part_uri, centroids, indices, and id_uri comprise the index.  The
 * partitioned database is stored in part_uri, the centroids are stored in
 * centroids, the indices demarcating partitions is stored in indices, and the
 * labels for the vectors in the original database are stored in id_uri.
 * The query is stored in q.
 *
 * "Infinite RAM" means the entire index is loaded into memory before any
 * queries are applied, regardless of which partitions are to be queried.
 *
 * @param partitioned_vectors Partitioned database
 * @param centroids Centroids of the vectors in the original database (and
 * the partitioned database).  The ith centroid is the centroid of the ith
 * partition.
 * @param q The query to be searched
 * @param indices The demarcations of partitions
 * @param partitioned_ids Labels for the vectors in the original database
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @return The indices of the top_k neighbors for each query vector
 *
 * @todo We should still order the queries so partitions are searched in order
 */
template <feature_vector_array F, query_vector_array Q, feature_vector_array C>
auto nuv_query_heap_infinite_ram_reg_blocked(
    const F& partitioned_vectors,
    const C& centroids,
    const Q& query,
    size_t nprobe,
    size_t k_nn,
    size_t nthreads) {
  if (num_loads(partitioned_vectors) == 0) {
    load(partitioned_vectors);
  }
  scoped_timer _{tdb_func__ + std::string{"_in_ram"}};

  using id_type = typename F::id_type;
  auto indices = partitioned_vectors.indices();
  auto partitioned_ids = partitioned_vectors.ids();

  using score_type = float;

  auto num_queries = num_vectors(query);

  // @todo Maybe we don't want to do new_indices in partition_ivf_index after
  //  all since they aren't used in this function
  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  // auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
  //     size(q), fixed_min_pair_heap<score_type, id_type>(k_nn));

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
             * For each partition, process the queries that have that
             * partition as their top centroid.
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
                  auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
                  auto score_01 = L2(q_vec_0, partitioned_vectors[kp + 1]);
                  auto score_10 = L2(q_vec_1, partitioned_vectors[kp + 0]);
                  auto score_11 = L2(q_vec_1, partitioned_vectors[kp + 1]);

                  min_scores[n][j0].insert(score_00, partitioned_ids[kp + 0]);
                  min_scores[n][j0].insert(score_01, partitioned_ids[kp + 1]);
                  min_scores[n][j1].insert(score_10, partitioned_ids[kp + 0]);
                  min_scores[n][j1].insert(score_11, partitioned_ids[kp + 1]);
                }

                /*
                 * Cleanup the last iteration(s) of k
                 */
                for (size_t kp = kstop; kp < stop; ++kp) {
                  auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
                  auto score_10 = L2(q_vec_1, partitioned_vectors[kp + 0]);
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
                  auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
                  auto score_01 = L2(q_vec_0, partitioned_vectors[kp + 1]);

                  min_scores[n][j0].insert(score_00, partitioned_ids[kp + 0]);
                  min_scores[n][j0].insert(score_01, partitioned_ids[kp + 1]);
                }
                for (size_t kp = kstop; kp < stop; ++kp) {
                  auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
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
  auto top_k = get_top_k_with_scores(min_scores, k_nn);

  return top_k;
}

/*******************************************************************************
 *
 * Functions for searching with infinite RAM
 *
 ******************************************************************************/

/**
 * @brief OG Implementation of finite RAM qv query.
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
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param upper_bound Limit of how many vectors to load into memory at one time
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @return The indices of the top_k neighbors for each query vector
 */
template <class feature_type, class id_type>
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
    size_t nthreads) {
  scoped_timer _{tdb_func__};

  using score_type = float;
  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;

  // Check that the size of the indices vector is correct
  assert(size(indices) == num_vectors(centroids) + 1);

  size_t num_queries = num_vectors(query);

  // get closest centroid for each query vector
  auto top_centroids =
      detail::flat::qv_query_heap_0(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(top_centroids)::value_type;

  /*
   * `top_centroids` maps from rank X query index to the centroid *index*.
   *
   * To process centroids (partitions) in order, we need to map from `centroid`
   * to the set of queries having that centroid.
   *
   * We also need to know the "active" centroids, i.e., the ones having at
   * least one query.
   */
  auto centroid_query = std::multimap<parts_type, id_type>{};
  auto active_centroids = std::set<parts_type>{};
  for (size_t j = 0; j < num_queries; ++j) {
    for (size_t p = 0; p < nprobe; ++p) {
      auto tmp = top_centroids(p, j);
      centroid_query.emplace(top_centroids(p, j), j);
      active_centroids.emplace(top_centroids(p, j));
    }
  }

  auto active_partitions =
      std::vector<parts_type>(begin(active_centroids), end(active_centroids));

  auto partitioned_vectors = tdbColMajorPartitionedMatrix<
      feature_type,
      id_type,
      indices_type,
      parts_type>(
      ctx, part_uri, indices, active_partitions, id_uri, upper_bound);

  std::vector<parts_type> new_indices(size(active_partitions) + 1);
  new_indices[0] = 0;
  for (size_t i = 0; i < size(active_partitions); ++i) {
    new_indices[i + 1] = new_indices[i] + indices[active_partitions[i] + 1] -
                         indices[active_partitions[i]];
  }

  {
    size_t max_partition_size{0};
    for (size_t i = 0; i < size(new_indices) - 1; ++i) {
      auto partition_size = new_indices[i + 1] - new_indices[i];
      max_partition_size = std::max<size_t>(max_partition_size, partition_size);
      _memory_data.insert_entry(
          tdb_func__ + " (predicted)",
          partition_size * sizeof(feature_type) *
              partitioned_vectors.num_rows());
    }
    _memory_data.insert_entry(
        tdb_func__ + " (upper bound)",
        nprobe * num_queries * sizeof(feature_type) * max_partition_size);
  }

  assert(partitioned_vectors.num_cols() == size(partitioned_vectors.ids()));

  debug_matrix(partitioned_vectors, "partitioned_vectors");
  debug_matrix(partitioned_vectors.ids(), "partitioned_vectors.ids()");

  // auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
  //       size(q), fixed_min_pair_heap<score_type, id_type>(k_nn));

  auto min_scores =
      std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>>(
          nthreads,
          std::vector<fixed_min_pair_heap<score_type, id_type>>(
              num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  log_timer _i{tdb_func__ + " in RAM"};

  while (partitioned_vectors.load()) {
    _i.start();

    auto current_part_size = partitioned_vectors.num_resident_parts();

    // size_t block_size = (size(active_partitions) + nthreads - 1) / nthreads;
    // size_t parts_per_thread =
    //        (size(active_partitions) + nthreads - 1) / nthreads;
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
            [&query,
             &min_scores,
             &partitioned_vectors,
             &new_indices,
             &centroid_query,
             &active_partitions,
             n,
             first_part,
             last_part]() {
              /*
               * For each partition, process the queries that have that
               * partition as their top centroid.
               */
              for (size_t p = first_part; p < last_part; ++p) {
                auto partno = p + resident_part_offset(partitioned_vectors);

                auto start = new_indices[partno];
                auto stop = new_indices[partno + 1];

                /*
                 * Get the queries associated with this partition.
                 */
                auto range =
                    centroid_query.equal_range(active_partitions[partno]);
                for (auto i = range.first; i != range.second; ++i) {
                  auto j = i->second;
                  auto q_vec = query[j];

                  // @todo shift start / stop back by the offset
                  for (size_t k = start; k < stop; ++k) {
                    auto kp = k - col_offset(partitioned_vectors);
                    auto score = L2(q_vec, partitioned_vectors[kp]);

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
  }

  consolidate_scores(min_scores);
  auto top_k = get_top_k_with_scores(min_scores, k_nn);

  return top_k;
}

// ----------------------------------------------------------------------------
// Functions for searching with finite RAM, new qv (nuv) ordering
// ----------------------------------------------------------------------------

template <class feature_type, class id_type>
auto nuv_query_heap_finite_ram(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& query,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads);

/**
 * Interface with uris for all arguments.
 */
template <
    typename feature_type,
    class id_type,
    class centroids_type,
    class indices_type>
auto nuv_query_heap_finite_ram(
    const std::string& part_uri,
    const std::string& centroids_uri,
    const std::string& query_uri,
    const std::string& indices_uri,
    const std::string& id_uri,
    size_t nqueries,
    size_t nprobe,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads) {
  tiledb::Context ctx;

  // using centroid_type =
  // std::invoke_result_t<tdbColMajorMatrix<centroids_type>>;
  using query_type = std::invoke_result_t<tdbColMajorMatrix<feature_type>>;
  using idx_type = std::invoke_result_t<tdbColMajorMatrix<indices_type>>;

  std::future<centroids_type> centroids_future =
      std::async(std::launch::async, [&]() {
        auto centroids = tdbColMajorMatrix<centroids_type>(ctx, centroids_uri);
        centroids.load();
        return centroids;
      });
  // auto centroids = tdbColMajorMatrix<centroids_type>(ctx, centroids_uri);
  // centroids.load();

  std::future<query_type> query_future = std::async(std::launch::async, [&]() {
    auto query =
        tdbColMajorMatrix<feature_type, id_type>(ctx, query_uri, nqueries);
    query.load();
    return query;
  });
  // auto query =
  //      tdbColMajorMatrix<db_type, id_type>(ctx, query_uri,
  //      nqueries);
  // query.load();

  std::future<idx_type> indices_future = std::async(std::launch::async, [&]() {
    auto indices = read_vector<indices_type>(ctx, indices_uri);
    return indices;
  });
  //  auto indices = read_vector<indices_type>(ctx, indices_uri);

  // Wait for completion in order of expected access time
  auto indices = indices_future.get();
  auto query = query_future.get();
  auto centroids = centroids_future.get();

  return nuv_query_heap_finite_ram(
      ctx,
      part_uri,
      centroids,
      query,
      indices,
      id_uri,
      nprobe,
      k_nn,
      upper_bound,
      nthreads);
}

/**
 * @brief OG Implementation of finite RAM using the new qv (nuv) ordering.
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
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param upper_bound Limit of how many vectors to load into memory at one time
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @return The indices of the top_k neighbors for each query vector
 */
template <class feature_type, class id_type>
auto nuv_query_heap_finite_ram(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& query,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads) {
  scoped_timer _{tdb_func__ + " " + part_uri};

  // Check that the size of the indices vector is correct
  assert(size(indices) == centroids.num_cols() + 1);

  using score_type = float;
  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;

  auto num_queries = num_vectors(query);

  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

  auto partitioned_vectors = tdbColMajorPartitionedMatrix<
      feature_type,
      id_type,
      indices_type,
      parts_type>(
      ctx, part_uri, indices, active_partitions, id_uri, upper_bound);

  std::vector<parts_type> new_indices(size(active_partitions) + 1);
  new_indices[0] = 0;
  for (size_t i = 0; i < size(active_partitions); ++i) {
    new_indices[i + 1] = new_indices[i] + indices[active_partitions[i] + 1] -
                         indices[active_partitions[i]];
  }

  {
    // Record some memory usage stats
    size_t max_partition_size{0};
    for (size_t i = 0; i < size(new_indices) - 1; ++i) {
      auto partition_size = new_indices[i + 1] - new_indices[i];
      max_partition_size = std::max<size_t>(max_partition_size, partition_size);
      _memory_data.insert_entry(
          tdb_func__ + " (predicted)",
          partition_size * sizeof(feature_type) *
              partitioned_vectors.num_rows());
    }
    _memory_data.insert_entry(
        tdb_func__ + " (upper bound)",
        nprobe * num_queries * sizeof(feature_type) * max_partition_size);
  }

  assert(partitioned_vectors.num_cols() == size(partitioned_vectors.ids()));
  debug_matrix(partitioned_vectors, "partitioned_vectors");
  debug_matrix(partitioned_vectors.ids(), "partitioned_vectors.ids()");

  // auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
  //       size(q), fixed_min_pair_heap<score_type, id_type>(k_nn));

  std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>> min_scores(
      nthreads,
      std::vector<fixed_min_pair_heap<score_type, id_type>>(
          num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  log_timer _i{tdb_func__ + " in RAM"};

  while (partitioned_vectors.load()) {
    _i.start();

    auto current_part_size = partitioned_vectors.num_resident_parts();

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
            [&, &active_queries = active_queries, n, first_part, last_part]() {
              /*
               * For each partition, process the queries that have that
               * partition as their top centroid.
               */
              for (size_t p = first_part; p < last_part; ++p) {
                auto partno = p + resident_part_offset(partitioned_vectors);
                auto start =
                    new_indices[partno] - col_offset(partitioned_vectors);
                auto stop =
                    new_indices[partno + 1] - col_offset(partitioned_vectors);
                ;

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
  }

  consolidate_scores(min_scores);
  auto top_k = get_top_k_with_scores(min_scores, k_nn);
  return top_k;
}

/**
 * @brief OG Implementation of finite RAM using the new qv (nuv) ordering.
 * In this function
 * we apply a tiling to the inner loop (similar to the approach used in
 * matrix-matrix product).
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
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param upper_bound Limit of how many vectors to load into memory at one time
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @return The indices of the top_k neighbors for each query vector
 */
template <class feature_type, class id_type>
auto nuv_query_heap_finite_ram_reg_blocked(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& query,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads) {
  scoped_timer _{tdb_func__ + " " + part_uri};

  // Check that the size of the indices vector is correct
  assert(size(indices) == centroids.num_cols() + 1);

  using score_type = float;
  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;

  auto num_queries = num_vectors(query);

  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

  auto partitioned_vectors = tdbColMajorPartitionedMatrix<
      feature_type,
      id_type,
      indices_type,
      parts_type>(
      ctx, part_uri, indices, active_partitions, id_uri, upper_bound);

  std::vector<parts_type> new_indices(size(active_partitions) + 1);
  new_indices[0] = 0;
  for (size_t i = 0; i < size(active_partitions); ++i) {
    new_indices[i + 1] = new_indices[i] + indices[active_partitions[i] + 1] -
                         indices[active_partitions[i]];
  }

  {
    // Record some memory usage stats
    size_t max_partition_size{0};
    for (size_t i = 0; i < size(new_indices) - 1; ++i) {
      auto partition_size = new_indices[i + 1] - new_indices[i];
      max_partition_size = std::max<size_t>(max_partition_size, partition_size);
      _memory_data.insert_entry(
          tdb_func__ + " (predicted)",
          partition_size * sizeof(feature_type) *
              partitioned_vectors.num_rows());
    }
    _memory_data.insert_entry(
        tdb_func__ + " (upper bound)",
        nprobe * num_queries * sizeof(feature_type) * max_partition_size);
  }

  assert(partitioned_vectors.num_cols() == size(partitioned_vectors.ids()));
  debug_matrix(partitioned_vectors, "partitioned_vectors");
  debug_matrix(partitioned_vectors.ids(), "partitioned_vectors.ids()");

  // auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
  //       size(q), fixed_min_pair_heap<score_type, id_type>(k_nn));

  std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>> min_scores(
      nthreads,
      std::vector<fixed_min_pair_heap<score_type, id_type>>(
          num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  log_timer _i{tdb_func__ + " in RAM"};

  while (partitioned_vectors.load()) {
    _i.start();

    auto current_part_size = partitioned_vectors.num_resident_parts();

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
             &new_indices,
             &active_queries = active_queries,
             &active_partitions = active_partitions,
             n,
             first_part,
             last_part]() {
              size_t col_offset = _cpo::col_offset(partitioned_vectors);
              size_t part_offset = resident_part_offset(partitioned_vectors);

              /*
               * For each partition, process the queries that have that
               * partition as their top centroid.
               */
              for (size_t p = first_part; p < last_part; ++p) {
                auto partno = p + part_offset;

                auto quartno = partno;
                //                if constexpr
                //                (!has_num_resident_parts<std::remove_cvref_t<decltype(partitioned_vectors)>>)
                //                {
                quartno = active_partitions[partno];
                //                }
                (void)active_partitions;  // Silence unused warning

                auto start = new_indices[quartno] - col_offset;
                auto stop = new_indices[quartno + 1] - col_offset;
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
                    auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
                    auto score_01 = L2(q_vec_0, partitioned_vectors[kp + 1]);
                    auto score_10 = L2(q_vec_1, partitioned_vectors[kp + 0]);
                    auto score_11 = L2(q_vec_1, partitioned_vectors[kp + 1]);

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
                    auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
                    auto score_10 = L2(q_vec_1, partitioned_vectors[kp + 0]);
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
                    auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
                    auto score_01 = L2(q_vec_0, partitioned_vectors[kp + 1]);

                    min_scores[n][j0].insert(
                        score_00, partitioned_vectors.ids()[kp + 0]);
                    min_scores[n][j0].insert(
                        score_01, partitioned_vectors.ids()[kp + 1]);
                  }
                  for (size_t kp = kstop; kp < stop; ++kp) {
                    auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
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
    _i.stop();
  }

  consolidate_scores(min_scores);
  auto top_k = get_top_k_with_scores(min_scores, k_nn);

  return top_k;
}

/**
 * @brief Component function for querying a portion of an IVF index.  It is
 * designed to be used in a parallel (and/or distributed) fashion.
 * @param query The entire set of queries being applied
 * @param partitioned_vectors The partitions loaded from the index. These are
 * the partitions that have at least one query associated with them.  Note that
 * we are loading partitions into contiguous memory, but they are not
 * necessarily contiguous in the original array. As a result, we can't use the
 * original indices to access the partitions.
 * @param new_indices Indices for demarcating partitions in partitioned_vectors.
 * @param active_queries For each centroid, the ids of the subset of queries
 * associated with that centroid.
 * @param ids The original labels of the vectors in the partitioned_vectors
 * @param active_partitions The set of partitions having at least one query
 * @param k_nn How many neigbors to return
 * @param first_part The first partition to query
 * @param last_part One past the last partition to query
 * @return
 */
template <query_vector_array Q, partitioned_feature_vector_array P>
auto apply_query(
    const Q& query,
    const P& partitioned_vectors,
    auto&& new_indices,
    auto&& active_queries,
    auto&& ids,
    auto&& active_partitions,
    size_t k_nn,
    size_t first_part,
    size_t last_part) {
  //  print_types(query, partitioned_vectors, new_indices, active_queries);

  // using feature_type = typename
  // std::remove_reference_t<decltype(partitioned_vectors)>::value_type;
  using id_type = typename std::remove_reference_t<decltype(ids)>::value_type;
  using score_type = float;

  auto num_queries = num_vectors(query);
  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn));

  size_t part_offset = 0;
  size_t col_offset = 0;
  col_offset = _cpo::col_offset(partitioned_vectors);
  part_offset = resident_part_offset(partitioned_vectors);

  for (size_t p = first_part; p < last_part; ++p) {
    auto partno = p + part_offset;

    // @todo this is a bit of a hack
    auto quartno = partno;
    //    if constexpr (!has_num_resident_parts<decltype(partitioned_vectors)>)
    //    {
    quartno = active_partitions[partno];
    //    }

    // We don't need col_offset fixup any longer
    auto start = new_indices[quartno];     // - col_offset;
    auto stop = new_indices[quartno + 1];  // - col_offset;

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
        auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
        auto score_01 = L2(q_vec_0, partitioned_vectors[kp + 1]);
        auto score_10 = L2(q_vec_1, partitioned_vectors[kp + 0]);
        auto score_11 = L2(q_vec_1, partitioned_vectors[kp + 1]);

        min_scores[j0].insert(score_00, ids[kp + 0]);
        min_scores[j0].insert(score_01, ids[kp + 1]);
        min_scores[j1].insert(score_10, ids[kp + 0]);
        min_scores[j1].insert(score_11, ids[kp + 1]);
      }

      /*
       * Cleanup the last iteration(s) of k
       */
      for (size_t kp = kstop; kp < stop; ++kp) {
        auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
        auto score_10 = L2(q_vec_1, partitioned_vectors[kp + 0]);
        min_scores[j0].insert(score_00, ids[kp + 0]);
        min_scores[j1].insert(score_10, ids[kp + 0]);
      }
    }

    /*
     * Cleanup the last iteration(s) of j
     */
    for (auto j = end; j < active_queries[partno].end(); ++j) {
      auto j0 = j[0];
      auto q_vec_0 = query[j0];

      for (size_t kp = start; kp < kstop; kp += 2) {
        auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
        auto score_01 = L2(q_vec_0, partitioned_vectors[kp + 1]);

        min_scores[j0].insert(score_00, ids[kp + 0]);
        min_scores[j0].insert(score_01, ids[kp + 1]);
      }
      for (size_t kp = kstop; kp < stop; ++kp) {
        auto score_00 = L2(q_vec_0, partitioned_vectors[kp + 0]);
        min_scores[j0].insert(score_00, ids[kp + 0]);
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
 * @param nprobe How many partitions to search
 * @param k_nn How many nearest neighbors to return
 * @param upper_bound Limit of how many vectors to load into memory at one time
 * @param nth Unused
 * @param nthreads How many threads to use for parallel execution
 * @param min_parts_per_thread Unused (WIP for threading heuristics)
 * @return The indices of the top_k neighbors for each query vector
 */
template <class feature_type, class id_type>
auto query_finite_ram(
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
    size_t min_parts_per_thread = 0) {
  scoped_timer _{tdb_func__ + " " + part_uri};

  // Check that the size of the indices vector is correct
  assert(size(indices) == centroids.num_cols() + 1);

  using score_type = float;
  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;

  auto num_queries = num_vectors(query);

  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

  auto partitioned_vectors = tdbColMajorPartitionedMatrix<
      feature_type,
      id_type,
      indices_type,
      parts_type>(
      ctx, part_uri, indices, active_partitions, id_uri, upper_bound);

  log_timer _i{tdb_func__ + " in RAM"};

  std::vector<parts_type> new_indices(size(active_partitions) + 1);
  new_indices[0] = 0;
  for (size_t i = 0; i < size(active_partitions); ++i) {
    new_indices[i + 1] = new_indices[i] + indices[active_partitions[i] + 1] -
                         indices[active_partitions[i]];
  }

  {
    // Record some memory usage stats
    size_t max_partition_size{0};
    for (size_t i = 0; i < size(new_indices) - 1; ++i) {
      auto partition_size = new_indices[i + 1] - new_indices[i];
      max_partition_size = std::max<size_t>(max_partition_size, partition_size);
      _memory_data.insert_entry(
          tdb_func__ + " (predicted)",
          partition_size * sizeof(feature_type) *
              partitioned_vectors.num_rows());
    }
    _memory_data.insert_entry(
        tdb_func__ + " (upper bound)",
        nprobe * num_queries * sizeof(feature_type) * max_partition_size);
  }

  assert(partitioned_vectors.num_cols() == size(partitioned_vectors.ids()));
  debug_matrix(partitioned_vectors, "partitioned_vectors");
  debug_matrix(partitioned_vectors.ids(), "partitioned_vectors.ids()");

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn));

  while (partitioned_vectors.load()) {
    _i.start();

    auto current_part_size = partitioned_vectors.num_resident_parts();

    size_t parts_per_thread = (current_part_size + nthreads - 1) / nthreads;

    {
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
               &new_indices,
               &active_queries = active_queries,
               &active_partitions = active_partitions,
               k_nn,
               first_part,
               last_part]() {
                return apply_query(
                    query,
                    partitioned_vectors,
                    new_indices,
                    active_queries,
                    partitioned_vectors.ids(),
                    active_partitions,
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
    }

    _i.stop();
  }

  auto top_k = get_top_k_with_scores(min_scores, k_nn);

  return top_k;
}

/**
 * @brief Reference implementation of infinite RAM query using qv ordering.
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
 * "Infinite RAM" means the entire partitioned database is loaded into RAM.
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
 * @param min_parts_per_thread Unused (WIP for threading heuristics)
 * @return The indices of the top_k neighbors for each query vector
 */
template <
    feature_vector_array F,
    feature_vector_array C,
    feature_vector_array Q>
auto query_infinite_ram(
    const F& partitioned_vectors,
    const C& centroids,
    const Q& query,
    size_t nprobe,
    size_t k_nn,
    size_t nthreads) {
  scoped_timer _{tdb_func__ + std::string{"_in_ram"}};

  auto partitioned_ids = partitioned_vectors.ids();
  auto indices = partitioned_vectors.indices();

  // using feature_type = typename
  // std::remove_reference_t<decltype(partitioned_vectors)>::value_type;
  using id_type =
      typename std::remove_reference_t<decltype(partitioned_ids)>::value_type;
  using score_type = float;

  // Check that the sizes of the index constituents are correct
  assert(num_vectors(partitioned_vectors) == partitioned_ids.size());
  assert(size(indices) == num_vectors(centroids) + 1);

  auto num_queries = num_vectors(query);

  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

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
           &indices,
           &active_queries = active_queries,
           &active_partitions = active_partitions,
           &partitioned_ids,
           k_nn,
           first_part,
           last_part]() {
            return apply_query(
                query,
                partitioned_vectors,
                indices,
                active_queries,
                partitioned_ids,
                active_partitions,
                k_nn,
                first_part,
                last_part);
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

  auto top_k = get_top_k_with_scores(min_scores, k_nn);

  return top_k;
}

template <class feature_type, class id_type>
auto query_infinite_ram(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    size_t nthreads) {
  scoped_timer _{tdb_func__};

  // Read the shuffled database and ids
  // @todo To this more systematically
  auto partitioned_vectors = tdbColMajorMatrix<feature_type>(ctx, part_uri);
  partitioned_vectors.load();
  auto partitioned_ids = read_vector<id_type>(ctx, id_uri);

  return query_infinite_ram(
      partitioned_vectors,
      centroids,
      q,
      indices,
      partitioned_ids,
      nprobe,
      k_nn,
      nthreads);
}

}  // namespace detail::ivf

#endif  // TILEDB_IVF_QV_H
