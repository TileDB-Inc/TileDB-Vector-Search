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
#include "algorithm.h"
#include "defs.h"
#include "linalg.h"
#include "timer.h"

// If apple, use Accelerate
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <mkl_cblas.h>
#endif

// Interfaces
//   faiss: D, I = index.search(xb, k) # search
//   milvus: status, results = conn.search(collection_name, query_records, top_k,
//   params) # search
//     "nlist" to create index (how many bins)
//     "nprobe" to search index (how many bins to search)

/**
 * @brief Query a single vector against a vector database.
 * Intended to be a high-level interface that can dispatch
 * to the right query function depending on the size of the query.
 * Not implemented at the moment.
 * @todo Implement this function
 *
 * @tparam Q Type of queries
 * @param db URI of the vector database
 * @param q Queries
 * @param k Number of nearest neighbors to return
 * @param nprobe Number of bins to search
 * @param nthreads Number of threads to use
 * @return tuple of distances and indices
 */
template <class Q>
auto kmeans_query(
    const std::string& db,
    const Q& q,
    size_t k,
    size_t nprobe,
    size_t nthreads) {
  // @todo: dispatch to the right query function depending on size of q
  //      (e.g., if q is a single vector, use query_single)
  //      (e.g., if q is a matrix, use query_batch)
  //      For now, just use query_single

  // Load centroids
  // Search nprobe of the centroids
  // Search the corresponding db partitions
}

/**
 * @brief Query a set of vectors agains a vector database.
 * @tparam DB Type of database (expected `Matrix`)
 * @tparam Q Type of queries (expected `Matrix`)
 * @param db Database
 * @param q Queries
 * @param k Number of nearest neighbors to return
 * @param nthreads Number of threads to use
 * @return tuple of indices of top k matches
 *
 * @todo Implement using parallel for_each.  Will need to add iterators etc
 * to the `Matrix` class to make this work.
 */
template <class DB, class Q>
auto qv_query(const DB& db, const Q& q, size_t k, unsigned nthreads) {
  life_timer _{"Total time (qv query)"};

  using element = std::pair<float, int>;

  Matrix<size_t> top_k(k, q.num_cols());

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

  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async, [k, start, stop, size_db, &q, &db, &top_k]() {
        for (size_t j = start; j < stop; ++j) {
          fixed_min_set<element> scores(k);
          size_t idx = 0;

          for (int i = 0; i < size_db; ++i) {
            auto score = L2(q[j], db[i]);
            scores.insert(element{score, i});
          }
          std::transform(
              scores.begin(), scores.end(), top_k[j].begin(), ([](auto&& e) {
                return e.second;
              }));
        }
          }));
    }
  }

  return top_k;
}

/**
 * @brief Query a single vector against a vector database, returning the
 * indices of the single best matches among the database vectors.
 *
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param nthreads
 * @return
 */
template <class DB, class Q>
auto qv_partition(const DB& db, const Q& q, unsigned nthreads) {
  life_timer _{"Total time (qv partition)"};

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

/**
 * @brief Query a set of vectors against a vector database, returning the
 * indices of all matches for each query vector.
 *
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param k
 * @param nthreads
 * @return
 */
template <class DB, class Q>
auto gemm_partition(const DB& db, const Q& q, unsigned nthreads) {
  life_timer _outer{"Total time gemm"};

  ColMajorMatrix<float> scores(db.num_cols(), q.num_cols());
  auto _score_data = raveled(scores);

  int M = db.num_cols();
  int N = q.num_cols();
  int K = db.num_rows();

  assert(db.num_rows() == q.num_rows());

  {
    life_timer _{"L2 comparison (gemm)"};

    cblas_sgemm(
        CblasColMajor,
        CblasTrans,
        CblasNoTrans,
        M,
        N,
        K,
        -2.0,
        &db(0, 0),
        K,
        &q(0, 0),
        K,
        0.0,
        &scores(0, 0),
        M);
  }

  std::vector<float> alpha(M, 0.0f);
  std::vector<float> beta(N, 0.0f);
  {
    life_timer _{"L2 comparison colsum"};

    mat_col_sum(db, alpha, [](auto a) { return a * a; });  // @todo optimize somehow
    mat_col_sum(q, beta, [](auto a) { return a * a; });
  }

  {
    life_timer _{"L2 comparison outer product"};

    // A += alpha * x * transpose(y)
    std::vector<float> alpha_ones(N, 1.0f);
    std::vector<float> beta_ones(M, 1.0f);

    // This should be more parallelizable -- but seems to be completely
    // memory-bound
    cblas_sger(
        CblasColMajor,
        M,
        N,
        1.0,
        &alpha[0],
        1,
        &alpha_ones[0],
        1,
        &scores(0, 0),
        M);
    cblas_sger(
        CblasColMajor,
        M,
        N,
        1.0,
        &beta_ones[0],
        1,
        &beta[0],
        1,
        &scores(0, 0),
        M);
  }

  {
    life_timer _{"L2 comparison finish"};

    stdx::execution::parallel_policy par{nthreads};
    stdx::for_each(
        std::move(par), begin(_score_data), end(_score_data), [](auto& a) {
          a = sqrt(a);
        });
  }

  auto top_k = std::vector<int>(q.num_cols());
  {
    life_timer _{"top k"};
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

#endif  // TDB_IVF_QUERY_H