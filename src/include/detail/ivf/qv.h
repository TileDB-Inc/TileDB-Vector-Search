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
 * operate on them blockwise, which ameliorates the locality issues that
 * arise when doing this order with a flat index, say.
 *
 * There are two implementations here: infinite RAM and finite RAM.  The
 * infinite RAM case loads the entire partitioned database into memory, and then
 * searches in the partitions as indicated by the nearest centroids to the
 * queries.  The infinite RAM case does not perform any out-of-core operations.
 * The finite RAM case only loads the partitions into memory that are necessary
 * for the search. The user can specify an upper bound on the amount of RAM to
 * be used for holding the queries being searched.  The searches are ordered so
 * that the partitions can be loaded into memory in the order they are layed out
 * in the array.
 *
 * In general there is probably no reason to ever use the infinite RAM case
 * other than for benchmarking, as it requires machines with very large amounts
 * of RAM.
 */

#ifndef TILEDB_IVF_QV_H
#define TILEDB_IVF_QV_H

#include <map>

#include "algorithm.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "flat_query.h"
#include "linalg.h"

namespace detail::ivf {

/**
 * Overload for already opened arrays.  Since the array is already opened, we
 * don't need to specify its type with a template parameter.
 */
auto qv_query_heap_infinite_ram(
    auto&& shuffled_db,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    auto&& shuffled_ids,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads);

/**
 * Overload for case where we need to open the vector and id arrays.  We can't
 * do any template argument deduction here because we need to know the type of
 * the vector array.
 */
template <typename T>
auto qv_query_heap_infinite_ram(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads);

/**
 * @brief Query a (small) set of query vectors against a vector database.
 * This version loads the entire partition array into memory and then
 * queries each vector in the query set against the appropriate partitions.
 *
 * For now that type of the array needs to be passed as a template argument.
 */
template <typename T, class shuffled_ids_type>
auto qv_query_heap_infinite_ram(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads) {
  scoped_timer _{tdb_func__};

  // Read the shuffled database and ids
  // @todo To this more systematically
  auto shuffled_db = tdbColMajorMatrix<T>(ctx, part_uri);
  auto shuffled_ids = read_vector<shuffled_ids_type>(ctx, id_uri);

  return qv_query_heap_infinite_ram(
      shuffled_db,
      centroids,
      q,
      indices,
      shuffled_ids,
      nprobe,
      k_nn,
      nth,
      nthreads);
}

auto qv_query_heap_infinite_ram(
    const std::string& part_uri,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads) {
  tiledb::Context ctx;
  return qv_query_heap_infinite_ram(
      ctx,
      part_uri,
      centroids,
      q,
      indices,
      id_uri,
      nprobe,
      k_nn,
      nth,
      nthreads);
}

// @todo We should still order the queries so partitions are searched in order
auto qv_query_heap_infinite_ram(
    auto&& shuffled_db,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    auto&& shuffled_ids,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads) {
  scoped_timer _{"Total time " + tdb_func__};

  assert(shuffled_db.num_cols() == shuffled_ids.size());

  // Check that the indices vector is the right size
  assert(size(indices) == centroids.num_cols() + 1);

  debug_matrix(shuffled_db, "shuffled_db");
  debug_matrix(shuffled_ids, "shuffled_ids");

  // get closest centroid for each query vector
  // auto top_k = qv_query(centroids, q, nprobe, nthreads);
  //  auto top_centroids = vq_query_heap(centroids, q, nprobe, nthreads);

  // @todo is this the best (fastest) algorithm to use?  (it takes miniscule
  // time at rate)
  auto top_centroids =
      detail::flat::qv_query_nth(centroids, q, nprobe, false, nthreads);

  auto min_scores = std::vector<fixed_min_pair_heap<float, size_t>>(
      size(q), fixed_min_pair_heap<float, size_t>(k_nn));

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
              auto score = L2(q_vec /*q[j]*/, shuffled_db[i]);
              min_scores[j].insert(score, shuffled_ids[i]);
            }
          }
        });
  }

  ColMajorMatrix<size_t> top_k(k_nn, q.num_cols());
  {
    scoped_timer ___{tdb_func__ + std::string{"_top_k"}};

    // get_top_k_from_heap(min_scores, top_k);

    // @todo get_top_k_from_heap
    for (int j = 0; j < size(q); ++j) {
      sort_heap(min_scores[j].begin(), min_scores[j].end());
      std::transform(
          min_scores[j].begin(),
          min_scores[j].end(),
          top_k[j].begin(),
          ([](auto&& e) { return std::get<1>(e); }));
    }
  }

  return top_k;
}

/**
 * @brief Query a set of vectors against a partitioned database.
 *
 * This function queries each vector in the query set against the appropriate
 * partitions.
 *
 * For now that type of the array and the type of the shuffled
 * ids need to be passed as template arguments.
 */
template <typename T, class shuffled_ids_type>
auto qv_query_heap_finite_ram(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    size_t upper_bound,
    bool nth,
    size_t nthreads) {
  scoped_timer _{tdb_func__};

  using parts_type =
      typename std::remove_reference_t<decltype(centroids)>::value_type;
  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;

  size_t num_queries = size(q);

  // get closest centroid for each query vector
  auto top_centroids =
      detail::flat::qv_query_nth(centroids, q, nprobe, false, nthreads);

  /*
   * `top_centroids` maps from rank X query index to the centroid index.
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
      centroid_query.emplace(top_centroids(p, j), j);
      active_centroids.emplace(top_centroids(p, j));
    }
  }

  // debug_slice(top_centroids, "top_centroids",
  // std::min<size_t>(16, top_centroids.num_rows()),
  // std::min<size_t>(16, top_centroids.num_cols()));

  auto active_partitions =
      std::vector<parts_type>(begin(active_centroids), end(active_centroids));

  // Check that the size of the indices vector is correct
  assert(size(indices) == centroids.num_cols() + 1);

  std::vector<parts_type> new_indices(size(active_partitions) + 1);
  new_indices[0] = 0;
  for (size_t i = 0; i < size(active_partitions); ++i) {
    new_indices[i + 1] = new_indices[i] + indices[active_partitions[i] + 1] -
                         indices[active_partitions[i]];
  }

  auto shuffled_db = tdbColMajorPartitionedMatrix<
      T,
      shuffled_ids_type,
      indices_type,
      parts_type>(
      ctx,
      part_uri,
      std::move(indices),
      active_partitions,
      id_uri,
      upper_bound,
      /* shuffled_ids,*/ nthreads);

  size_t max_partition_size{0};
  for (size_t i = 0; i < size(new_indices) - 1; ++i) {
    auto partition_size = new_indices[i + 1] - new_indices[i];
    max_partition_size = std::max<size_t>(max_partition_size, partition_size);
    _memory_data.insert_entry(
        tdb_func__ + " (predicted)",
        partition_size * sizeof(T) * shuffled_db.num_rows());
  }
  _memory_data.insert_entry(
      tdb_func__ + " (upper bound)",
      nprobe * num_queries * sizeof(T) * max_partition_size);

  assert(shuffled_db.num_cols() == size(shuffled_db.ids()));

  debug_matrix(shuffled_db, "shuffled_db");
  debug_matrix(shuffled_db.ids(), "shuffled_db.ids()");

  // auto min_scores = std::vector<fixed_min_pair_heap<float, size_t>>(
  //       size(q), fixed_min_pair_heap<float, size_t>(k_nn));

  std::vector<std::vector<fixed_min_pair_heap<float, size_t>>> min_scores(
      nthreads,
      std::vector<fixed_min_pair_heap<float, size_t>>(
          size(q), fixed_min_pair_heap<float, size_t>(k_nn)));

  log_timer _i{tdb_func__ + " in RAM"};

  while(shuffled_db.load()) {

    _i.start();

    // size_t block_size = (size(active_partitions) + nthreads - 1) / nthreads;
    size_t parts_per_thread =
        (shuffled_db.num_col_parts() + nthreads - 1) / nthreads;

    std::vector<std::future<void>> futs;
    futs.reserve(nthreads);

    for (size_t n = 0; n < nthreads; ++n) {
      auto first_part =
          std::min<size_t>(n * parts_per_thread, shuffled_db.num_col_parts());
      auto last_part = std::min<size_t>(
          (n + 1) * parts_per_thread, shuffled_db.num_col_parts());

      if (first_part != last_part) {
        futs.emplace_back(
            std::async(std::launch::async, [&, n, first_part, last_part]() {
              /*
               * For each partition, process the queries that have that
               * partition as their top centroid.
               */
              for (size_t p = first_part; p < last_part; ++p) {
                auto partno = p + shuffled_db.col_part_offset();
                auto start = new_indices[partno];
                auto stop = new_indices[partno + 1];

                /*
                 * Get the queries associated with this partition.
                 */
                auto range =
                    centroid_query.equal_range(active_partitions[partno]);
                for (auto i = range.first; i != range.second; ++i) {
                  auto j = i->second;
                  auto q_vec = q[j];

                  // @todo shift start / stop back by the offset
                  for (size_t k = start; k < stop; ++k) {
                    auto kp = k - shuffled_db.col_offset();

                    auto score = L2(q_vec, shuffled_db[kp]);

                    // @todo any performance with apparent extra indirection?
                    min_scores[n][j].insert(score, shuffled_db.ids()[kp]);
                  }
                }
              }
            }));
      }
    }

    for (int n = 0; n < size(futs); ++n) {
      futs[n].get();
    }
    _i.stop();
  }

  _i.start();
  for (int j = 0; j < size(q); ++j) {
    for (int n = 1; n < nthreads; ++n) {
      for (auto&& e : min_scores[n][j]) {
        min_scores[0][j].insert(std::get<0>(e), std::get<1>(e));
      }
    }
  }
  _i.stop();


  scoped_timer ___{tdb_func__ + std::string{"_top_k"}};

  ColMajorMatrix<size_t> top_k(k_nn, q.num_cols());

  // get_top_k_from_heap(min_scores, top_k);

  // @todo get_top_k_from_heap
  for (int j = 0; j < size(q); ++j) {
    sort_heap(min_scores[0][j].begin(), min_scores[0][j].end());
    std::transform(
        min_scores[0][j].begin(),
        min_scores[0][j].end(),
        top_k[j].begin(),
        ([](auto&& e) { return std::get<1>(e); }));
  }

  return top_k;
}


}  // namespace detail::ivf

#endif  // TILEDB_IVF_QV_H
