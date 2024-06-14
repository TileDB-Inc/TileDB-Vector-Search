/**
 * @file   vq.h
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
 * A set of algorithms that use the inverted file index (IVF) for querying. The
 * set of vectors to search are provided in partitioned form, where each
 * partition consists of the vectors closest to a given centroid vector (the
 * centroids are also provided).  In addition to the query vectors to be
 * applied, an IVF search also specifies how many partitions to search (`nprobe`
 * or the number of centroids to probe).
 *
 * To perform a search using the index, given a query vector `q`
 * * Find the `nprobe` centroids closest to `q[i]`, using a top k query of q
 * against the centroids, with k = `nprobe`.  These centroids are the "active
 * centroids" and the corresponding partitions are the "active partitions".
 * * Determine the set of queries that are associated with each of the active
 * partitions (i.e., the queries for which each centroid is in the top `nprobe`
 * closest centroids for the query
 * * For each active partition, search the vectors in the partition for the
 * closest matches to the queries
 * * Return `top_k`, but using the indexes for the vectors (ids) corresponding
 * to their locations in the original (unpartitioned) data set
 *
 * Note that the same process applies to the case of an "infinite RAM" query
 * or a "finite RAM" query.  In the former case, the entire database is loaded
 * into memory and the designated active partitions are searched. In the latter
 * case, only the active partitions are loaded into memory and searched.  In
 * addition, in the finite RAM case, the vectors to be searched may be loaded
 * in out-of-core fashion.
 *
 * The actual query application in `vq_apply_query`, loops over vectors on the
 * outer loop and queries on the inner loop.
 * To perform a search using the index, given a query vector `q[i]`
 *
 */

#ifndef TILEDB_IVF_VQ_H
#define TILEDB_IVF_VQ_H

#include "partition.h"
#include "scoring.h"

namespace detail::ivf {

/**
 * @brief Apply a query to a set of partitioned vectors.  In order to support
 * the infinite RAM and finite RAM cases, the input may
 * consist of all of the partitions in the database or a subset of them (only
 * the ones that need to be searched).  In the former case, the set of
 * partitions to be searched is given by the active_partitions vector.  In the
 * latter case, the active_partitions vector is ignored.
 *
 * Note that this algorithm is essentially the transpose of the one in qv.h.
 *
 * @param query The set of all query vectors.
 * @param partitioned_db The partitioned set of vectors to be searched
 * @param new_indices The indices delimiting the partitions.
 * @param active_queries Indicates which queries to apply to each of the active
 * partitions.
 * @param ids The ids of the vectors in the database.
 * @param active_partitions The active partitions.
 * @param k_nn The number of nearest neighbors to return.
 * @param first_part The first partition to search.
 * @param last_part The last partition to search.
 *
 * @return A vector of pairs of scores and ids.
 */
auto vq_apply_query(
    auto&& query,
    auto&& partitioned_db,
    auto&& new_indices,
    auto&& active_queries,
    auto&& ids,
    auto&& active_partitions,
    size_t k_nn,
    size_t first_part,
    size_t last_part) {
  auto num_queries = num_vectors(query);

  // using feature_type = typename
  // std::remove_reference_t<decltype(partitioned_db)>::value_type;
  using id_type = typename std::remove_reference_t<decltype(ids)>::value_type;
  using score_type = float;

  size_t col_offset = _cpo::col_offset(partitioned_db);
  size_t part_offset = col_part_offset(partitioned_db);

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn));

  /**
   * Loop over given partitons to be searched
   */
  for (size_t p = first_part; p < last_part; ++p) {
    auto partno = p + part_offset;

    // @todo this is a bit of a hack
    auto quartno = partno;
    //    if constexpr (!has_num_col_parts<decltype(partitioned_db)>) {
    quartno = active_partitions[partno];
    //    }

    auto start = new_indices[quartno] - col_offset;
    auto stop = new_indices[quartno + 1] - col_offset;
    auto kstep = stop - start;
    auto kstop = start + 2 * (kstep / 2);

    // size_t i_begin = ii, i_step = std::min<size_t>(64, ii_last - ii), i_end =
    // i_begin + 2 * (i_step / 2), i_last = i_begin + i_step;

    auto len = 2 * (size(active_queries[partno]) / 2);
    auto end = active_queries[partno].begin() + len;

    /*
     * Loop over the vectors in the partition
     */
    for (size_t kp = start; kp < kstop; kp += 2) {
      /*
       * Loop over the queries associated with the partition
       */
      for (auto j = active_queries[partno].begin(); j < end; j += 2) {
        auto j0 = j[0];
        auto j1 = j[1];
        auto q_vec_0 = query[j0];
        auto q_vec_1 = query[j1];

        auto score_00 = L2(q_vec_0, partitioned_db[kp + 0]);
        auto score_01 = L2(q_vec_0, partitioned_db[kp + 1]);
        auto score_10 = L2(q_vec_1, partitioned_db[kp + 0]);
        auto score_11 = L2(q_vec_1, partitioned_db[kp + 1]);

        min_scores[j0].insert(score_00, ids[kp + 0]);
        min_scores[j0].insert(score_01, ids[kp + 1]);
        min_scores[j1].insert(score_10, ids[kp + 0]);
        min_scores[j1].insert(score_11, ids[kp + 1]);
      }

      /*
       * Cleanup the last iteration(s) of j
       */
      for (auto j = end; j < active_queries[partno].end(); ++j) {
        auto j0 = j[0];
        auto q_vec_0 = query[j0];

        auto score_00 = L2(q_vec_0, partitioned_db[kp + 0]);
        auto score_01 = L2(q_vec_0, partitioned_db[kp + 1]);
        min_scores[j0].insert(score_00, ids[kp + 0]);
        min_scores[j0].insert(score_01, ids[kp + 1]);
      }
    }

    /*
     * Cleanup the last iteration(s) of k
     */
    for (size_t kp = kstop; kp < stop; ++kp) {
      for (auto j = active_queries[partno].begin(); j != end; j += 2) {
        auto j0 = j[0];
        auto j1 = j[1];
        auto q_vec_0 = query[j0];
        auto q_vec_1 = query[j1];

        auto score_00 = L2(q_vec_0, partitioned_db[kp + 0]);
        auto score_10 = L2(q_vec_1, partitioned_db[kp + 0]);

        min_scores[j0].insert(score_00, ids[kp + 0]);
        min_scores[j1].insert(score_10, ids[kp + 0]);
      }

      /*
       * Cleanup the last last iteration(s) of j
       */
      for (auto j = end; j < active_queries[partno].end(); ++j) {
        auto j0 = j[0];
        auto q_vec_0 = query[j0];

        auto score_00 = L2(q_vec_0, partitioned_db[kp + 0]);
        min_scores[j0].insert(score_00, ids[kp + 0]);
      }
    }
  }
  return min_scores;
}

/**
 * Similar to vq_query_finite_ram, but the entire database is loaded into RAM.
 * This function takes a partitioned matrix as input, which is assumed to
 * already have loaded all of its data.
 */
auto vq_query_infinite_ram(
    auto&& partitioned_db,
    auto&& centroids,
    auto&& query,
    auto&& indices,
    auto&& partitioned_ids,
    size_t nprobe,
    size_t k_nn,
    size_t nthreads) {
  scoped_timer _{tdb_func__ + std::string{"_in_ram"}};

  // using feature_type = typename
  // std::remove_reference_t<decltype(partitioned_db)>::value_type;
  using id_type =
      typename std::remove_reference_t<decltype(partitioned_ids)>::value_type;
  using score_type = float;

  if (partitioned_db.num_cols() != partitioned_ids.size()) {
    throw std::runtime_error(
        "[vq@vq_query_infinite_ram] partitioned_db.num_cols() != "
        "partitioned_ids.size()");
  }

  // Check that the indices vector is the right size
  if (size(indices) != centroids.num_cols() + 1) {
    throw std::runtime_error(
        "[vq@vq_query_infinite_ram] size(indices) != centroids.num_cols() + 1");
  }

  auto num_queries = num_vectors(query);

  // @todo Maybe we don't want to do new_indices in partition_ivf_index after
  //  all since they aren't used in this function
  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

  std::vector<parts_type> new_indices(size(active_partitions) + 1);

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
           &partitioned_db,
           &indices,
           &active_queries = active_queries,
           &active_partitions = active_partitions,
           &partitioned_ids,
           k_nn,
           first_part,
           last_part]() {
            return vq_apply_query(
                query,
                partitioned_db,
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

  return get_top_k_with_scores(min_scores, k_nn);
}

/**
 * Function that takes a URI to a partitioned matrix and a query matrix,
 * loads them each into a matrix, and then calls `vq_query_infinite_ram`
 * above.
 */
template <class feature_type, class id_type>
auto vq_query_infinite_ram(
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

  // Read the partitioned database and ids
  // @todo To this more systematically
  auto partitioned_db = tdbColMajorMatrix<feature_type>(ctx, part_uri);
  partitioned_db.load();
  auto partitioned_ids = read_vector<id_type>(ctx, id_uri);

  return vq_query_infinite_ram(
      partitioned_db,
      centroids,
      q,
      indices,
      partitioned_ids,
      nprobe,
      k_nn,
      nthreads);
}

/**
 * Similar to vq_query_finite_ram, but the entire database is loaded into RAM.
 * This function takes a partitioned matrix as input, which is assumed to
 * already have loaded all of its data.
 */
auto vq_query_infinite_ram_2(
    auto&& partitioned_db,
    auto&& centroids,
    auto&& query,
    auto&& indices,
    auto&& partitioned_ids,
    size_t nprobe,
    size_t k_nn,
    size_t nthreads) {
  scoped_timer _{tdb_func__ + std::string{"_in_ram"}};

  if (partitioned_db.num_cols() != partitioned_ids.size()) {
    throw std::runtime_error(
        "[vq@vq_query_infinite_ram_2] partitioned_db.num_cols() != "
        "partitioned_ids.size()");
  }

  // Check that the indices vector is the right size
  if (size(indices) != centroids.num_cols() + 1) {
    throw std::runtime_error(
        "[vq@vq_query_infinite_ram_2] size(indices) != centroids.num_cols() + "
        "1");
  }

  // using feature_type = typename
  // std::remove_reference<decltype(partitioned_db)>::value_type;
  using id_type =
      typename std::remove_reference<decltype(partitioned_ids)>::value_type;
  using score_type = float;

  auto num_queries = num_vectors(query);

  // @todo Maybe we don't want to do new_indices in partition_ivf_index after
  //  all since they aren't used in this function
  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

  std::vector<parts_type> new_indices(size(active_partitions) + 1);

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
           &min_scores,
           &partitioned_db,
           &new_indices = indices,
           &active_queries = active_queries,
           &active_partitions = active_partitions,
           n,
           first_part,
           last_part]() {
            /*
             * For each partition, process the queries that have that
             * partition as their top centroid.
             */
            for (size_t partno = first_part; partno < last_part; ++partno) {
              auto quartno = active_partitions[partno];
              auto start = new_indices[quartno];
              auto stop = new_indices[quartno + 1];

              /*
               * Get the queries associated with this partition.
               */

              for (size_t k = start; k < stop; ++k) {
                auto kp = k - partitioned_db.col_offset();

                for (auto j : active_queries[partno]) {
                  // @todo shift start / stop back by the offset

                  auto score = L2(query[j], partitioned_db[kp]);

                  // @todo any performance with apparent extra indirection?
                  min_scores[j].insert(score, partitioned_db.ids()[kp]);
                }
              }
            }
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

/**
 * Function that takes a URI to a partitioned matrix and a query matrix,
 * loads them each into a matrix, and then calls `vq_query_infinite_ram`
 * above.
 */
template <typename T, class id_type>
auto vq_query_infinite_ram_2(
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

  // Read the partitioned database and ids
  // @todo To this more systematically
  auto partitioned_db = tdbColMajorMatrix<T>(ctx, part_uri);
  partitioned_db.load();
  auto partitioned_ids = read_vector<id_type>(ctx, id_uri);

  return vq_query_infinite_ram(
      partitioned_db,
      centroids,
      q,
      indices,
      partitioned_ids,
      nprobe,
      k_nn,
      nthreads);
}

/*
 * @brief Perform a query on a partitioned database, using `vq_apply_query`
 * This function determines the active partitions and active queries, and
 * then creates a partitioned matrix holding the vectors from the identified
 * active partitions.  (This may be done in out of core fashion, so that
 * only a subset of the active partitions are loaded at any one time.)  The
 * function then invoked `vq_apply_query` on the partitioned matrix in
 * parallel fashion, decomposing over the partitions.
 */
template <class feature_type, class id_type>
auto vq_query_finite_ram(
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
  if (size(indices) != centroids.num_cols() + 1) {
    throw std::runtime_error(
        "[vq@vq_query_finite_ram] size(indices) != centroids.num_cols() + 1");
  }

  using score_type = float;

  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;

  auto num_queries = num_vectors(query);

  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

  auto partitioned_db = tdbColMajorPartitionedMatrix<
      feature_type,
      id_type,
      indices_type,
      parts_type>(
      ctx, part_uri, indices, active_partitions, id_uri, upper_bound);
  load(partitioned_db);

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
          partition_size * sizeof(feature_type) * partitioned_db.num_rows());
    }
    _memory_data.insert_entry(
        tdb_func__ + " (upper bound)",
        nprobe * num_queries * sizeof(feature_type) * max_partition_size);
  }

  if (partitioned_db.num_cols() != size(partitioned_db.ids())) {
    throw std::runtime_error(
        "[vq@vq_query_finite_ram] partitioned_db.num_cols() != "
        "size(partitioned_db.ids())");
  }

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn));

  do {
    _i.start();

    auto current_part_size = partitioned_db.num_col_parts();

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
               &partitioned_db,
               &new_indices,
               &active_queries = active_queries,
               &active_partitions = active_partitions,
               k_nn,
               first_part,
               last_part]() {
                return vq_apply_query(
                    query,
                    partitioned_db,
                    new_indices,
                    active_queries,
                    partitioned_db.ids(),
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
          for (auto&& e : min_n[j]) {
            min_scores[j].insert(std::get<0>(e), std::get<1>(e));
          }
        }
      }
    }

    _i.stop();
  } while (load(partitioned_db));

  return get_top_k_with_scores(min_scores, k_nn);
}

template <class feature_type, class id_type>
auto vq_query_finite_ram_2(
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

  using score_type = float;
  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;

  auto num_queries = num_vectors(query);

  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

  auto partitioned_db = tdbColMajorPartitionedMatrix<
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

  auto min_scores =
      std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>>(
          nthreads,
          std::vector<fixed_min_pair_heap<score_type, id_type>>(
              num_queries, fixed_min_pair_heap<score_type, id_type>(k_nn)));

  while (partitioned_db.load()) {
    _i.start();

    auto current_part_size = partitioned_db.num_col_parts();

    size_t parts_per_thread = (current_part_size + nthreads - 1) / nthreads;

    {
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
               &partitioned_db,
               &new_indices,
               &active_queries = active_queries,
               n,
               first_part,
               last_part]() {
                /*
                 * For each partition, process the queries that have that
                 * partition as their top centroid.
                 */
                for (size_t p = first_part; p < last_part; ++p) {
                  auto partno = p + partitioned_db.col_part_offset();

                  auto start = new_indices[partno];
                  auto stop = new_indices[partno + 1];

                  /*
                   * Get the queries associated with this partition.
                   */

                  for (size_t k = start; k < stop; ++k) {
                    auto kp = k - partitioned_db.col_offset();

                    for (auto j : active_queries[partno]) {
                      // @todo shift start / stop back by the offset

                      auto score = L2(query[j], partitioned_db[kp]);

                      // @todo any performance with apparent extra indirection?
                      min_scores[n][j].insert(score, partitioned_db.ids()[kp]);
                    }
                  }
                }
              }));
        }
      }
    }

    _i.stop();
  }

  consolidate_scores(min_scores);
  return get_top_k_with_scores(min_scores, k_nn);
}

}  // namespace detail::ivf

#endif  // TILEDB_IVF_VQ_H
