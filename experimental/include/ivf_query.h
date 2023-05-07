/**
 * @file   ivf_query.h
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
 * Contains some basic query functions for kmeans indexing.
 *
 */

#ifndef TDB_IVF_QUERY_H
#define TDB_IVF_QUERY_H

#include <algorithm>
#include <chrono>
#include "algorithm.h"
#include <chrono>
#include "algorithm.h"
#include "concepts.h"
#include "defs.h"
#include "flat_query.h"
#include "linalg.h"
#include "scoring.h"
#include "utils/timer.h"


#ifndef tdb_func__
#ifdef __cpp_lib_source_location
#include <source_location>
#define tdb_func__ (std::source_location::current().function_name())
#else
#define tdb_func__ std::string{(__func__)}
#endif
#endif

// If apple, use Accelerate
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <mkl_cblas.h>
#endif

// Interfaces
//   faiss: D, I = index.search(xb, k) # search
//   milvus: status, results = conn.search(collection_name, query_records,
//   top_k, params) # search
//     "nlist" to create index (how many bins)
//     "nprobe" to search index (how many bins to search)

/**
 * @brief Query a single vector against a vector database.
 * Intended to be a high-level interface that can dispatch
 * to the right query function depending on the size of the query.
 */

auto kmeans_query(
    const std::string& part_uri,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    auto&& shuffled_ids,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads) {
#if 1
    return kmeans_query_small_q(
        part_uri,
        centroids,
        q,
        indices,
        shuffled_ids,
        nprobe,
        k_nn,
        nth,
        nthreads);

#else
  if (q.size() < 5) {
    return kmeans_query_small_q(
        shuffled_db,
        centroids,
        q,
        indices,
        shuffled_ids,
        nprobe,
        k_nn,
        nth,
        nthreads);
  } else {
    return kmeans_query_large_q(
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
  #endif
}

/**
 * @brief Query a set of query vectors against a vector database.
 *
 * This will need to be restructured to support blocking.
 */
auto kmeans_query_large_q(
    auto&& shuffled_db,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    auto&& shuffled_ids,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads) {
  // get closest centroid for each query vector
  // Does this even need to be blocked...?
  // The whole point of ivf is to avoid loading everything
  // The shuffled_db is the big array to avoid loading
  auto top_k = blocked_gemm_query(centroids, q, nprobe, nth, nthreads);

  // Copy top k from Matrix to vector
  std::vector<size_t> top_top_k(nprobe, 0);
  for (size_t i = 0; i < nprobe; ++i) {
    top_top_k[i] = top_k(i, 0);
  }

  // gather all the probed partitions into a single matrix
  size_t total_size = 0;
  for (size_t i = 0; i < size(top_top_k); ++i) {
    total_size += indices[top_top_k[i] + 1] - indices[top_top_k[i]];
  }

  // Storage for the probed partitions and their ids
  auto all_results = ColMajorMatrix<float>{centroids.num_rows(), total_size};
  auto all_ids = std::vector<uint64_t>(total_size);

  // Tracks next location to copy into
  size_t ctr = 0;

  // @todo parallelize this loop
  // @todo don't make contiguous copy -- just search each cluster separately
  // Copy the probed partitions into contiguous storage
  // For each probed partition
  for (size_t j = 0; j < nprobe; ++j) {
    // Get begin and end indices of the partition
    size_t start = indices[top_top_k[j]];
    size_t end = indices[top_top_k[j] + 1];

    // Copy the partition into the storage
    // For each vector in the partition
    for (size_t i = start; i < end; ++i) {
      // Copy the vector into all_results and ids into all_ids
      // @todo Abstract all of this explicit loop based assignment
      size_t l_end = shuffled_db.num_rows();
      for (size_t l = 0; l < l_end; ++l) {
        all_results(l, ctr) = shuffled_db(l, i);
        all_ids[ctr] = shuffled_ids[i];
      }
      ++ctr;
    }
  }

  // Now, with the single matrix of probed partitions, find the closest vectors
  auto kmeans_ids = blocked_gemm_query(all_results, q, k_nn, nth, nthreads);

  // Original ids are: all_ids[kmeans_ids(i, 0)]
  // Maybe that is what should be returned?

  return std::make_tuple(std::move(kmeans_ids), all_ids);
}

/**
 * @brief Query a (small) set of query vectors against a vector database.
 */
auto kmeans_query_small_q(
    const std::string& part_uri,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    const std::string& id_uri,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads) {

  // get closest centroid for each query vector
  // auto top_k = qv_query(centroids, q, nprobe, nthreads);
  //  auto top_centroids = vq_query_heap(centroids, q, nprobe, nthreads);
  auto top_centroids = qv_query_nth(centroids, q, nprobe, false, nthreads);

  auto shuffled_db = tdbColMajorMatrix<shuffled_db_type>(part_uri);
  auto shuffled_ids = read_vector<shuffled_ids_type>(id_uri);

  debug_matrix(shuffled_db, "shuffled_db");
  debug_matrix(shuffled_ids, "shuffled_ids");

  using element = std::pair<float, uint64_t>;
  auto min_scores =  std::vector<fixed_min_heap<element>>(size(q), fixed_min_heap<element>(k_nn));

#if 0
  for (size_t j = 0; j < size(q); ++j) {
    for (size_t p = 0; p < nprobe; ++p) {

      size_t start = indices[top_centroids(p, j)];
      size_t stop = indices[top_centroids(p, j) + 1];

      for (size_t i = start; i < stop; ++i) {
        auto score = L2(q[j], shuffled_db[i]);
        min_scores[j].insert(element{score, shuffled_ids[i]});
      }
    }
  }
#else

        life_timer __{std::string{"In memory portion of "} + tdb_func__};
        auto par = stdx::execution::indexed_parallel_policy{nthreads};
        stdx::range_for_each(
                std::move(par), q, [&, nprobe](auto &&q_vec, auto &&n = 0, auto &&j = 0) {
                    for (size_t p = 0; p < nprobe; ++p) {

                        size_t start = indices[top_centroids(p, j)];
                        size_t stop = indices[top_centroids(p, j) + 1];

                        for (size_t i = start; i < stop; ++i) {
                            auto score = L2(q[j], shuffled_db[i]);
                            min_scores[j].insert(element{score, shuffled_ids[i]});
                        }
                    }

                });
#endif

        ColMajorMatrix<size_t> top_k(k_nn, q.num_cols());

        for (int j = 0; j < size(q); ++j) {
            sort_heap(min_scores[j].begin(), min_scores[j].end());
            std::transform(
                    min_scores[j].begin(),
                    min_scores[j].end(),
                    top_k[j].begin(),
                    ([](auto &&e) { return e.second; }));
        }
    

  return top_k;
}

/**
 * @todo Block the query to avoid memory blowup
 *
 * @note qv_query_by_vector in flat_query.h is similar to this, but
 * uses foreach instead of manually spawning threads.
 */
template <vector_database DB, class Q>
auto qv_query(const DB& db, const Q& q, size_t k, unsigned nthreads) {
  life_timer _{"Total time " + tdb_func__};

  using element = std::pair<float, int>;

  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  // Have to do explicit asynchronous threading here, as the current parallel
  // algorithms have iterator-based interaces, and the `Matrix` class does not
  // yet have iterators.
  // @todo Implement iterator interface to `Matrix` class
  size_t size_db = db.num_cols();
  size_t size_q = q.num_cols();
  size_t container_size = size_q;
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  // @todo: Use range::for_each
  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async, [k, start, stop, size_db, &q, &db, &top_k]() {
            for (size_t j = start; j < stop; ++j) {
              fixed_min_set<element> min_scores(k);
              size_t idx = 0;

              for (int i = 0; i < size_db; ++i) {
                auto score = L2(q[j], db[i]);
                min_scores.insert(element{score, i});
              }
              std::sort(min_scores.begin(), min_scores.end());
              std::transform(
                  min_scores.begin(),
                  min_scores.end(),
                  top_k[j].begin(),
                  ([](auto&& e) { return e.second; }));
            }
          }));
    }
  }

  for (int n = 0; n < size(futs); ++n) {
    futs[n].get();
  }

  return top_k;
}

template <class DB, class Q>
auto qv_partition(const DB& db, const Q& q, unsigned nthreads) {
  life_timer _{"Total time " + tdb_func__};

  // Just need a single vector
  std::vector<unsigned> top_k(q.num_cols());

  // Again, doing the parallelization by hand here....
  size_t size_db = db.num_cols();
  size_t size_q = q.num_cols();
  size_t container_size = size_q;
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async, [start, stop, size_db, &q, &db, &top_k]() {
            for (size_t j = start; j < stop; ++j) {
              float min_score = std::numeric_limits<float>::max();
              size_t idx = 0;

              for (int i = 0; i < size_db; ++i) {
                auto score = L2(q[j], db[i]);
                if (score < min_score) {
                  min_score = score;
                  idx = i;
                }
              }

              top_k[j] = idx;
            }
          }));
    }
  }
  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].wait();
  }
  return top_k;
}

template <class DB, class Q>
auto gemm_query(const DB& db, const Q& q, int k, bool nth, size_t nthreads) {
  life_timer _{"Total time " + tdb_func__};
  auto scores = gemm_scores(db, q, nthreads);
  auto top_k = get_top_k(scores, k, nth, nthreads);
  return top_k;
}

using namespace std::chrono_literals;

template <class DB, class Q>
auto blocked_gemm_query(DB& db, Q& q, int k, bool nth, size_t nthreads) {
  life_timer _{"Total time " + tdb_func__};

  using element = std::pair<float, unsigned>;

  // @todo constexpr block_db and block_q
  auto block_db = db.is_blocked();
  auto block_q = q.is_blocked();
  auto async_db = block_db && db.is_async();
  auto async_q = block_q && q.is_async();
  if (block_db && block_q) {
    throw std::runtime_error("Can't block both db and q");
  }

  ColMajorMatrix<float> scores(db.num_cols(), q.num_cols());

  std::vector<fixed_min_heap<element>> min_scores(
      size(q), fixed_min_heap<element>(k));

  for (;;) {
    if (async_db) {
      db.advance_async();
    }
    if (async_q) {
      q.advance_async();
    }
    gemm_scores(db, q, scores, nthreads);

    auto par = stdx::execution::indexed_parallel_policy{nthreads};
    stdx::range_for_each(
        std::move(par), scores, [&](auto&& q_vec, auto&& n = 0, auto&& i = 0) {
          if (block_db) {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i].insert({scores(j, i), j + db.offset()});
            }
          } else if (block_q) {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i + q.offset()].insert({scores(j, i), j});
            }
          } else {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i].insert({scores(j, i), j});
            }
          }
        });

    bool done = true;
    if (block_db) {
      done = async_db ? !db.advance_wait() : !db.advance();
    } else if (block_q) {
      done = async_q ? !q.advance_wait() : !q.advance();
    }
    if (done) {
      break;
    }
  }

  ColMajorMatrix<size_t> top_k(k, q.num_cols());
  for (int j = 0; j < min_scores.size(); ++j) {
    // @todo sort_heap
    std::sort_heap(min_scores[j].begin(), min_scores[j].end());
    std::transform(
        min_scores[j].begin(),
        min_scores[j].end(),
        top_k[j].begin(),
        ([](auto&& e) { return e.second; }));
  }

  return top_k;
}

template <class DB, class Q>
auto gemm_partition(const DB& db, const Q& q, unsigned nthreads) {
  life_timer _{"Total time " + tdb_func__};

  auto scores = gemm_scores(db, q, nthreads);

  auto top_k = std::vector<int>(q.num_cols());
  {
    for (int i = 0; i < scores.num_cols(); ++i) {
      auto min_score = std::numeric_limits<float>::max();
      auto idx = 0;

      for (int j = 0; j < scores.num_rows(); ++j) {
        auto score = scores(j, i);
        if (score < min_score) {
          min_score = score;
          idx = j;
        }
      }
      top_k[i] = idx;
    }
  }

  return top_k;
}

template <class DB, class Q>
auto blocked_gemm_partition(DB& db, Q& q, unsigned nthreads) {
  life_timer _{"Total time " + tdb_func__};

  const auto block_db = db.is_blocked();
  const auto block_q = q.is_blocked();
  if (block_db && block_q) {
    throw std::runtime_error("Can't block both db and q");
  }

  ColMajorMatrix<float> scores(db.num_cols(), q.num_cols());
  auto _score_data = raveled(scores);
  auto top_k = std::vector<int>(q.num_cols());
  auto min_scores =
      std::vector<float>(q.num_cols(), std::numeric_limits<float>::max());

  for (;;) {
    gemm_scores(db, q, scores, nthreads);

    for (int i = 0; i < scores.num_cols(); ++i) {
      auto min_score = min_scores[i];
      auto idx = db.offset();

      for (int j = 0; j < scores.num_rows(); ++j) {
        auto score = scores(j, i);
        if (score < min_score) {
          min_score = score;
          idx = j + db.offset();
        }
      }
      top_k[i] = idx;
    }
    bool done = true;
    if (block_db) {
      done = !db.advance();
    } else {
      done = !q.advance();
    }
    if (done) {
      break;
    }
  }
  return top_k;
}

#endif  // TDB_IVF_QUERY_H
