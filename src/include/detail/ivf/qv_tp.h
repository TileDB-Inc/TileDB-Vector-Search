
#ifndef TILEDB_IVF_QV_TP_H
#define TILEDB_IVF_QV_TP_H

#include <string>
#include "stats.h"
#include "concepts.h"
#include "utils/fixed_min_queues.h"
#include "utils/threadpool.h"
#include "detail/linalg/matrix.h"

#include <tiledb/tiledb>

namespace detail::ivf {
auto apply_query_tp(
    auto&& query,
    auto&& shuffled_db,
    auto&& new_indices,
    auto&& active_queries,
    auto&& ids,
    auto&& active_partitions,
    size_t k_nn,
    size_t first_part,
    size_t last_part,
    auto&& min_scores) {
  auto num_queries = size(query);
  auto n = threadpool.get_thread_id();

  // std::cout << "thread " << n << " running " << first_part << " to " << last_part << std::endl;

  size_t part_offset = 0;
  size_t col_offset = 0;
  if constexpr (has_num_col_parts<decltype(shuffled_db)>) {
    part_offset = shuffled_db.col_part_offset();
    col_offset = shuffled_db.col_offset();
  }

  for (size_t p = first_part; p < last_part; ++p) {
    auto partno = p + part_offset;

    // @todo this is a bit of a hack
    auto quartno = partno;
    if constexpr (!has_num_col_parts<decltype(shuffled_db)>) {
      quartno = active_partitions[partno];
    }

    auto start = new_indices[quartno] - col_offset;
    auto stop = new_indices[quartno + 1] - col_offset;

    auto len = 2 * (size(active_queries[partno]) / 2);
    auto end = active_queries[partno].begin() + len;
    for (auto j = active_queries[partno].begin(); j != end; j += 2) {
      auto j0 = j[0];
      auto j1 = j[1];
      auto q_vec_0 = query[j0];
      auto q_vec_1 = query[j1];

      auto kstop = std::min<size_t>(stop, 2 * (stop / 2));
      for (size_t kp = start; kp < kstop; kp += 2) {
        auto score_00 = L2(q_vec_0, shuffled_db[kp + 0]);
        auto score_01 = L2(q_vec_0, shuffled_db[kp + 1]);
        auto score_10 = L2(q_vec_1, shuffled_db[kp + 0]);
        auto score_11 = L2(q_vec_1, shuffled_db[kp + 1]);

        min_scores[n][j0].insert(score_00, ids[kp + 0]);
        min_scores[n][j0].insert(score_01, ids[kp + 1]);
        min_scores[n][j1].insert(score_10, ids[kp + 0]);
        min_scores[n][j1].insert(score_11, ids[kp + 1]);
      }

      /*
       * Cleanup the last iteration(s) of k
       */
      for (size_t kp = kstop; kp < kstop; ++kp) {
        auto score_00 = L2(q_vec_0, shuffled_db[kp + 0]);
        auto score_10 = L2(q_vec_1, shuffled_db[kp + 0]);
        min_scores[n][j0].insert(score_00, ids[kp + 0]);
        min_scores[n][j1].insert(score_10, ids[kp + 0]);
      }
    }

    /*
     * Cleanup the last iteration(s) of j
     */
    for (auto j = end; j < active_queries[partno].end(); ++j) {
      auto j0 = j[0];
      auto q_vec_0 = query[j0];

      auto kstop = std::min<size_t>(stop, 2 * (stop / 2));
      for (size_t kp = start; kp < kstop; kp += 2) {
        auto score_00 = L2(q_vec_0, shuffled_db[kp + 0]);
        auto score_01 = L2(q_vec_0, shuffled_db[kp + 1]);

        min_scores[n][j0].insert(score_00, ids[kp + 0]);
        min_scores[n][j0].insert(score_01, ids[kp + 1]);
      }

      /*
       * Cleanup the last last iteration(s) of k
       */
      for (size_t kp = kstop; kp < stop; ++kp) {
        auto score_00 = L2(q_vec_0, shuffled_db[kp + 0]);
        min_scores[n][j0].insert(score_00, ids[kp + 0]);
      }
    }
  }
}

template <typename T, class shuffled_ids_type>
auto query_finite_ram_tp(
    tiledb::Context& ctx,
    const std::string& part_uri,
    auto&& centroids,
    auto&& query,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    size_t upper_bound,
    bool nth,
    size_t nthreads,
    size_t min_parts_per_thread = 0) {
  scoped_timer _{tdb_func__ + " " + part_uri};

  // Check that the size of the indices vector is correct
  assert(size(indices) == centroids.num_cols() + 1);

  std::cout << "nthreads = " << nthreads << ", " << threadpool.num_threads() << "\n";

  using indices_type =
      typename std::remove_reference_t<decltype(indices)>::value_type;

  auto num_queries = size(query);

  auto&& [active_partitions, active_queries] =
      partition_ivf_index(centroids, query, nprobe, nthreads);

  using parts_type = typename decltype(active_partitions)::value_type;

  auto shuffled_db = tdbColMajorPartitionedMatrix<
      T,
      shuffled_ids_type,
      indices_type,
      parts_type>(
      ctx, part_uri, indices, active_partitions, id_uri, upper_bound);

  std::vector<parts_type> new_indices(size(active_partitions) + 1);
  new_indices[0] = 0;
  for (size_t i = 0; i < size(active_partitions); ++i) {
    new_indices[i + 1] = new_indices[i] + indices[active_partitions[i] + 1] -
                         indices[active_partitions[i]];
  }


  ColMajorMatrix<size_t> top_k(k_nn, num_queries);
  auto min_scores =
      std::vector<std::vector<fixed_min_pair_heap<float, size_t>>>(
          nthreads,
          std::vector<fixed_min_pair_heap<float, size_t>>(
              num_queries, fixed_min_pair_heap<float, size_t>(k_nn)));

  log_timer _i{tdb_func__ + " in RAM"};

  while (shuffled_db.load()) {
    _i.start();

    auto current_part_size = shuffled_db.num_col_parts();

    {
      std::vector<std::future<void>> futs;
      futs.reserve(current_part_size);

      for (size_t n = 0; n < current_part_size; ++n) {
        auto first_part = n;
        auto last_part = n + 1;

        if (first_part != last_part) {
          futs.emplace_back(
              threadpool.async([&query,
                                &shuffled_db,
                                &new_indices,
                                &active_queries = active_queries,
                                &active_partitions = active_partitions,
                                &min_scores,
                                k_nn,
                                first_part,
                                last_part
                                ]() {
                return apply_query_tp(
                    query,
                    shuffled_db,
                    new_indices,
                    active_queries,
                    shuffled_db.ids(),
                    active_partitions,
                    k_nn,
                    first_part,
                    last_part,
                    min_scores);
              }));
        }
      }

      for (size_t n = 0; n < size(futs); ++n) {
        futs[n].wait();
      }
    }

    _i.stop();
  }

  scoped_timer ___{tdb_func__ + std::string{"_top_k"}};

  for (size_t j = 0; j < num_queries; ++j) {
    for (size_t n = 1; n < nthreads; ++n) {
      for (auto&& e : min_scores[n][j]) {
        min_scores[0][j].insert(std::get<0>(e), std::get<1>(e));
      }
    }
  }

  // @todo get_top_k_from_heap
  for (size_t j = 0; j < num_queries; ++j) {
    sort_heap(min_scores[0][j].begin(), min_scores[0][j].end());
    std::transform(
          min_scores[0][j].begin(),
          min_scores[0][j].end(),
          top_k[j].begin(),
          ([](auto&& e) { return std::get<1>(e); }));
  }

  return top_k;
}
} // namespace detail::ivf
#endif // TILEDB_IVF_QV_TP_H