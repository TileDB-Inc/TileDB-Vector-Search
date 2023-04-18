//
// Created by Andrew Lumsdaine on 4/17/23.
//

#ifndef TDB_QUERY_H
#define TDB_QUERY_H

#include "defs.h"
#include "timer.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

// If apple, use Accelerate
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <mkl_cblas.h>
#endif


template <class DB, class Q, class G, class TK>
void query_qv(const DB& db, const Q&q, const G& g, TK& top_k, int k, bool hardway) {
  if (hardway) {
    query_qv_hw(db, q, g, top_k, k);
  } else {
    query_qv_ew(db, q, g, top_k, k);
  }
}

template <class DB, class Q, class G, class TK>
void query_vq(const DB& db, const Q&q, const G& g, TK& top_k, int k, bool hardway) {
  if (hardway) {
    query_vq_hw(db, q, g, top_k, k);
  } else {
    query_vq_ew(db, q, g, top_k, k);
  }
}

template <class DB, class Q, class G, class TK>
void query_qv_hw(const DB& db, const Q&q, const G& g, TK& top_k, int k) {
  life_timer _{"Total time (vq hard way)"};

#pragma omp parallel
  {
    std::vector<int> i_index(size(db));
    std::vector<float> scores(size(db));

    std::iota(begin(i_index), end(i_index), 0);
    std::vector<int> index(size(db));

#pragma omp for
    // For each query
    for (size_t j = 0; j < size(q); ++j) {

      // Compare with each database vector
      for (size_t i = 0; i < size(db); ++i) {
        scores[i] = L2(q[j], db[i]);
      }

      // APPLE clang does not support execution policies
      std::copy(/*std::execution::seq,*/ begin(i_index), end(i_index), begin(index));
      get_top_k(scores, top_k[j], index, k);
      verify_top_k(scores, top_k[j], g[j], k, j);
    }
  }
}

template <class DB, class Q, class G, class TK>
void query_qv_ew(const DB& db, const Q&q, const G& g, TK& top_k, int k) {
  life_timer _{"Total time (eq set way)"};

  using element = std::pair<float, int>;

#pragma omp parallel for
  // For each query vector
  for (size_t j = 0; j < size(q); ++j) {

    // Create a set of the top k scores
    fixed_min_set<element> scores(k);

    // Compare with each database vector
    for (size_t i = 0; i < size(db); ++i) {
      auto score = L2(q[j], db[i]);
      scores.insert(element{score, i});
    }

    // Copy indexes into top_k
    std::transform(scores.begin(), scores.end(), top_k[j].begin(), ([](auto &e) { return e.second; }));

    // Try to break ties by sorting the top k
    std::sort(begin(top_k[j]), end(top_k[j]));
    std::sort(begin(g[j]), begin(g[j]) + k);
    verify_top_k(top_k[j], g[j], k, j);
  }
}


template <class DB, class Q, class G, class TK>
void query_vq_hw(const DB& db, const Q&q, const G& g, TK& top_k, int k) {
  life_timer _{"Total time (vq loop nesting, hard way)"};

  std::vector<std::vector<float>> scores(size(q), std::vector<float>(size(db), 0.0f));

  {
    life_timer _{"L2 distance"};

#pragma omp parallel for
    // For each database vector
    for (size_t i = 0; i < size(db); ++i) {

      // Compare with each query
      for (size_t j = 0; j < size(q); ++j) {
        scores[j][i] = L2(q[j], db[i]);
      }
    }
  }

  {
    life_timer _{"Get top k"};

#pragma omp parallel
    {
      std::vector<int> i_index(size(db));
      std::iota(begin(i_index), end(i_index), 0);
      std::vector<int> index(size(db));
#pragma omp for
      for (size_t j = 0; j < size(q); ++j) {
        std::copy(begin(i_index), end(i_index), begin(index));
        get_top_k(scores[j], top_k[j], index, k);
      }
    }
  }

  {
    life_timer _{"Checking results"};

#pragma omp parallel for
    for (size_t j = 0; j < size(q); ++j) {
      verify_top_k(scores[j], top_k[j], g[j], k, j);
    }
  }
}

template <class DB, class Q, class G, class TK>
void query_vq_ew(const DB& db, const Q&q, const G& g, TK& top_k, int k) {
  life_timer _{"Total time (vq loop nesting, set way)"};

  using element = std::pair<float, int>;
  std::vector<fixed_min_set<element>> scores(size(q), fixed_min_set<element>(k));

  {
    life_timer _{"L2 distance"};

    // For each database vector
    for (size_t i = 0; i < size(db); ++i) {

      // Can't parallelize outer loop b/c there is only one scores vector
#pragma omp parallel for

      // Compare with each query
      for (size_t j = 0; j < size(q); ++j) {
        auto score = L2(q[j], db[i]);
        scores[j].insert(element{score, i});
      }
    }
  }

  {
    life_timer _{"Get top k"};

#pragma omp parallel for
    for (size_t j = 0; j < size(q); ++j) {
      std::transform(scores[j].begin(), scores[j].end(), top_k[j].begin(), ([](auto &e) { return e.second; }));
      std::sort(begin(top_k[j]), end(top_k[j]));
    }
  }

  {
    life_timer _{"Checking results"};
#pragma omp parallel for
    for (size_t j = 0; j < size(q); ++j) {
      std::sort(begin(g[j]), begin(g[j]) + k);
      verify_top_k(top_k[j], g[j], k, j);
    }
  }
}

template <class DB, class Q, class G, class TK>
void query_gemm(const DB& db, const Q&q, const G& g, TK& top_k, int k, bool hardway) {

  /**
       * scores is nsamples X nq
       * db is dimension X nsamples
       * q is vsize X dimension
       * scores <- db^T * q
       */
  ms_timer init_time("Allocating score array");
  init_time.start();

  std::vector<std::span<float>> scores(size(q));

#if __APPLE__
  std::vector<float> _score_data(size(q) * size(db));
#else
  auto buf = std::make_unique_for_overwrite<float[]>(size(q)*size(db));
  std::span<float> _score_data {buf.get(), size(q)*size(db)};
#endif
  init_time.stop();
  std::cout << init_time << std::endl;
#ifdef __APPLE__
  std::cout << "Apple clang does not yet support make_unique_for_overwrite" << std::endl;
  std::cout << "so this is about 3X slower than it should be" << std::endl;
#endif

  size_t M = size(db);
  size_t N = size(q);
  size_t K = size(db[0]);
  assert(size(db[0]) == size(q[0]));

  /**
   * Compute the score matrix, based on (a - b)^2 = a^2 + b^2 - 2ab
   * scores[j][i] = alpha[i] + beta[j] - 2 * db[i] * q[j]
   */

  // Each score[j] is a column of the score matrix
  for (size_t j = 0; j < size(q); ++j) {
    scores[j] = std::span<float>(_score_data.data() + j * M, M);
  }

  // It may save some time to do this first
  {
    life_timer _{"L2 comparison (gemm)"};

    cblas_sgemm(
            CblasColMajor,
            CblasTrans,  // db^T
            CblasNoTrans,// q
            (int32_t) M, // number of samples
            (int32_t) N, // number of queries
            (int32_t) K, // dimension of vectors
            -2.0,
            db[0].data(),// A: K x M -> A^T: M x K
            K,
            q[0].data(),// B: K x N
            K,
            0.0,
            _score_data.data(),// C: M x N
            M);
  }

  std::vector<float> alpha(M, 0.0f);
  std::vector<float> beta(N, 0.0f);

  {
    life_timer _{"L2 comparison colsum"};

    col_sum(db, alpha, [](auto a) { return a * a; });
    col_sum(q, beta, [](auto a) { return a * a; });
  }

  {
    life_timer _{"L2 comparison outer product"};

    // A += alpha * x * transpose(y)
    std::vector<float> alpha_ones(N, 1.0f);
    std::vector<float> beta_ones(M, 1.0f);

    cblas_sger(CblasColMajor, M, N, 1.0, &alpha[0], 1, &alpha_ones[0], 1, _score_data.data(), M);
    cblas_sger(CblasColMajor, M, N, 1.0, &beta_ones[0], 1, &beta[0], 1, _score_data.data(), M);
  }

  {
    life_timer _{"L2 comparison finish"};

#if 0
    // APPLE clang does not support std execution policies
    std::for_each(/*std::execution::par_unseq,*/ begin(_score_data), end(_score_data), [](auto &&x) {
      x = sqrt(x);
    });
#else
#pragma omp parallel for
    for (size_t i = 0; i < size(_score_data); ++i) {
      _score_data[i] = sqrt(_score_data[i]);
    }
#endif
  }

  {
    life_timer _{"Get top k"};

    std::vector<int> i_index(size(db));
    std::iota(begin(i_index), end(i_index), 0);

#pragma omp parallel
    {
      std::vector<int> index(size(db));

#pragma omp for
      for (size_t j = 0; j < size(q); ++j) {
        std::copy(/*std::execution::seq,*/ begin(i_index), end(i_index), begin(index));
        get_top_k(scores[j], top_k[j], index, k);
      }
    }
  }

  {
    life_timer _{"Checking results"};
#pragma omp parallel for
    for (size_t j = 0; j < size(q); ++j) {
      verify_top_k(scores[j], top_k[j], g[j], k, j);
    }
  }
}

#endif//TDB_QUERY_H
