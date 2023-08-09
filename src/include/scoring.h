/**
 * @file   scoring.h
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
 * Gemm-based scoring.
 *
 */

#ifndef TDB_SCORING_H
#define TDB_SCORING_H

#include <algorithm>
#include "algorithm.h"
#include "concepts.h"
#include "linalg.h"
#include "utils/timer.h"

#include "detail/linalg/choose_blas.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <span>
// #include <execution>

#include "linalg.h"
#include "utils/fixed_min_queues.h"
#include "utils/timer.h"


// ----------------------------------------------------------------------------
// Distance functions
// ----------------------------------------------------------------------------

/**
 * @brief Compute sum of squares distance between two vectors.
 * @tparam V
 * @tparam U
 * @param a
 * @param b
 * @return
 */
#if 0
template <class V, class U>
inline auto sum_of_squares(V const& a, U const& b) {
  float sum{0.0};
  size_t size_a = size(a);

  if constexpr (std::is_same_v<decltype(a[0]),decltype(b[0])>) {
    for (size_t i = 0; i < size_a; ++i) {
      float diff = a[i]- b[i];
      sum += diff * diff;
    }
  } else {
    for (size_t i = 0; i < size_a; ++i) {
      float diff = ((float)a[i]) - ((float)b[i]);
      sum += diff * diff;
    }
  }
  return sum;
}
#else
template <class V, class U>
inline auto sum_of_squares(V const& a, U const& b) {
  float sum{0.0};
  size_t size_a = size(a);

  for (size_t i = 0; i < size_a; ++i) {
    // float diff = (float)a[i] - (float)b[i];  // converting to float is slow
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}
#endif

/**
 * @brief Compute L2 distance between two vectors.
 * @tparam V
 * @param a
 * @param b
 * @return L2 norm of the difference between a and b.
 */
template <class V, class U>
inline auto L2(V const& a, U const& b) {
  // return std::sqrt(sum_of_squares(a, b)); // sqrt is really slow
  return sum_of_squares(a, b);
}

/**
 * @brief Compute cosine similarity between two vectors.
 * @tparam V
 * @param a
 * @param b
 * @return
 */
template <class V>
auto cosine(V const& a, V const& b) {
  float sum = 0.0;
  float a2 = 0.0;
  float b2 = 0.0;

  auto size_a = size(a);
  for (auto i = 0; i < size_a; ++i) {
    sum += a[i] * b[i];
    a2 += a[i] * a[i];
    b2 += b[i] * b[i];
  }
  // return sum / std::sqrt(a2 * b2);  // sqrt is really slow
  return (sum * sum) / (a2 * b2);
}

/**
 * @brief Compute cosine similarity between two vectors.
 * @tparam V
 * @param a
 * @param b
 * @return
 */
template <class U, class V>
inline auto dot(U const& a, V const& b) {
  float sum = 0.0;

  auto size_a = size(a);
  for (auto i = 0; i < size_a; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}


// ----------------------------------------------------------------------------
// Functions for extracting top k neighbors from a raw scores matrix
// ----------------------------------------------------------------------------

// @todo implement with fixed_min_heap
template <class V, class L, class I>
auto get_top_k_nth(V const& scores, L&& top_k, I& index, int k) {
  std::iota(begin(index), end(index), 0);
  std::nth_element(
      begin(index), begin(index) + k, end(index), [&](auto&& a, auto&& b) {
        return scores[a] < scores[b];
      });
  std::copy(begin(index), begin(index) + k, begin(top_k));
  std::sort(begin(top_k), end(top_k), [&](auto& a, auto& b) {
    return scores[a] < scores[b];
  });
  return top_k;
}

template <class V, class L>
auto get_top_k(V const& scores, L&& top_k, int k) {
  fixed_min_pair_heap<float, unsigned> s(k);

  auto num_scores = scores.size();
  for (size_t i = 0; i < num_scores; ++i) {
    s.insert(scores[i], i);
  }
  get_top_k_from_heap(s, top_k);

  return top_k;
}

template <class S>
auto get_top_k(const S& scores, int k, bool nth, int nthreads) {
  scoped_timer _{"Get top k"};

  auto num_queries = scores.num_cols();

  auto top_k = ColMajorMatrix<size_t>(k, num_queries);

  int q_block_size = (num_queries + nthreads - 1) / nthreads;
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (int n = 0; n < nthreads; ++n) {
    int q_start = n * q_block_size;
    int q_stop = std::min<int>((n + 1) * q_block_size, num_queries);

    if (nth) {
      futs.emplace_back(std::async(
          std::launch::async, [q_start, q_stop, &scores, &top_k, k]() {
            std::vector<int> index(scores.num_rows());

            for (int j = q_start; j < q_stop; ++j) {
              get_top_k_nth(scores[j], std::move(top_k[j]), index, k);
            }
          }));
    } else {
      futs.emplace_back(std::async(
          std::launch::async, [q_start, q_stop, &scores, &top_k, k]() {
            std::vector<int> index(scores.num_rows());

            for (int j = q_start; j < q_stop; ++j) {
              get_top_k(scores[j], std::move(top_k[j]), k);
            }
          }));
    }
  }
  for (int n = 0; n < nthreads; ++n) {
    futs[n].get();
  }
  return top_k;
}

// ----------------------------------------------------------------------------
// Functions for extracting top k neighbors from a min heap (with pairs)
// ----------------------------------------------------------------------------

/**
 * @brief Utility function to put the top scores for multiple threads into a
 * single top_scores vector (the zeroth vector).
 * @tparam Heap
 *   @param min_scores a vector of vectors of min_heaps.  Each vector of min_heaps
 * is the top k scores for a set of queries.  Each vector of vectors is stores
 * a vector of min_heaps, one per thread.
 */
template <class Heap>
void consolidate_scores(std::vector<std::vector<Heap>>& min_scores) {
  auto nthreads = size(min_scores);
  auto num_queries = size(min_scores[0]);
  for (size_t j = 0; j < num_queries; ++j) {
    for (size_t n = 1; n < nthreads; ++n) {
      for (auto&& [e, f] : min_scores[n][j]) {
        min_scores[0][j].insert(e, f);
      }
    }
  }
}

/**
 * @brief Utility function to extract the top k scores from a single min heap.
 * @param min_scores
 * @param top_k
 */
inline void get_top_k_from_heap(auto&& min_scores, auto&& top_k) {
  std::sort_heap(begin(min_scores), end(min_scores));
  std::transform(
      begin(min_scores), end(min_scores), begin(top_k), ([](auto&& e) {
        return std::get<1>(e);
      }));
}

/**
 * @brief Utility function to extract the top k scores from a vector of min
 * heaps.  Each entry in the vector of min heaps is the top k scores for a
 * single query.
 * @tparam Heap
 * @tparam Index
 * @param scores
 * @param k_nn
 * @return a Matrix of the top_k scores for each query.  Each column corresponds
 * to a query,
 */
template <class Heap, class Index = size_t>
inline auto get_top_k(std::vector<Heap>& scores, size_t k_nn) {
  auto num_queries = size(scores);

  ColMajorMatrix<Index> top_k(k_nn, num_queries);

  for (size_t j = 0; j < num_queries; ++j) {
    get_top_k_from_heap(scores[j], top_k[j]);
  }
  return top_k;
}


/**
 * @brief Utility function to extract the top k scores from a vector of vectors.
 * It is assumed that the scores have been consolidated, i.e., that the zeroth
 * vector contains the top k scores for each query.
 * @tparam Heap
 * @tparam Index
 * @param scores
 * @param k_nn
 * @return Matrix of the top k scores for each query.  Each column corresponds
 * to a query.
 */
template <class Heap, class Index = size_t>
inline auto get_top_k(std::vector<std::vector<Heap>>& scores, size_t k_nn) {
  return get_top_k(scores[0], k_nn);
}


// ----------------------------------------------------------------------------
// Functions for computing top k neighbors with scores
// ----------------------------------------------------------------------------

inline void get_top_k_with_scores_from_heap(auto&& min_scores, auto&& top_k, auto&& top_k_scores) {
  std::sort_heap(begin(min_scores), end(min_scores));
  std::transform(
      begin(min_scores), end(min_scores), begin(top_k_scores), ([](auto&& e) {
        return std::get<0>(e);
      }));
  std::transform(
      begin(min_scores), end(min_scores), begin(top_k), ([](auto&& e) {
        return std::get<1>(e);
      }));
}

// Overload for one-d scores
template <class Heap, class Index = size_t>
inline auto get_top_k_with_scores(std::vector<Heap>& scores, size_t k_nn) {
  auto num_queries = size(scores);

  using score_type = typename Heap::value_type::first_type;

  ColMajorMatrix<Index> top_k(k_nn, num_queries);
  ColMajorMatrix<score_type> top_scores(k_nn, num_queries);

  for (size_t j = 0; j < num_queries; ++j) {
    get_top_k_with_scores_from_heap(scores[j], top_k[j]);
  }
  return top_k;
}

// Overload for two-d scores
template <class Heap, class Index = size_t>
inline auto get_top_k_with_scores(std::vector<std::vector<Heap>>& scores, size_t k_nn) {
  return get_top_k_with_scores(scores[0], k_nn);
}


// ----------------------------------------------------------------------------
// Functions for verifying top k neighbors against groundtruth
// ----------------------------------------------------------------------------


template <class L, class I>
auto verify_top_k_index(L const& top_k, I const& g, int k, int qno) {
  // std::sort(begin(g), begin(g) + k);
  // std::sort(begin(top_k), end(top_k));

  if (!std::equal(begin(top_k), begin(top_k) + k, g.begin())) {
    std::cout << "Query " << qno << " is incorrect" << std::endl;
    for (int i = 0; i < std::min<int>(10, k); ++i) {
      std::cout << "(" << top_k[i] << " != " << g[i] << ")  ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
}

/**
 * @brief Check the computed top k vectors against the ground truth.
 * Useful only for exact search.
 * Prints diagnostic message if difference is found.
 * @todo Handle the error more systematically and succinctly.
 */
template <class V, class L, class I>
auto verify_top_k(V const& scores, L const& top_k, I const& g, int k, int qno) {
  if (!std::equal(
          begin(top_k), begin(top_k) + k, g.begin(), [&](auto& a, auto& b) {
            return scores[a] == scores[b];
          })) {
    std::cout << "Query " << qno << " is incorrect" << std::endl;
    for (int i = 0; i < std::min<int>(10, k); ++i) {
      std::cout << "  (" << top_k[i] << " " << scores[top_k[i]] << ") ";
    }
    std::cout << std::endl;
    for (int i = 0; i < std::min(10, k); ++i) {
      std::cout << "  (" << g[i] << " " << scores[g[i]] << ") ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
}

/**
 * @brief Check the computed top k vectors against the ground truth.
 * Useful only for exact search.
 */
template <class L, class I>
auto verify_top_k(L const& top_k, I const& g, int k, int qno) {
  if (!std::equal(begin(top_k), begin(top_k) + k, g.begin())) {
    std::cout << "Query " << qno << " is incorrect" << std::endl;
    for (int i = 0; i < std::min(k, 10); ++i) {
      std::cout << "  (" << top_k[i] << " " << g[i] << ")";
    }
    std::cout << std::endl;
  }
}



template <class TK, class G>
bool validate_top_k(TK& top_k, G& g) {
  size_t k = top_k.num_rows();
  size_t num_errors = 0;

  for (size_t qno = 0; qno < top_k.num_cols(); ++qno) {
    // @todo -- count intersections rather than testing for equality
    std::sort(begin(top_k[qno]), end(top_k[qno]));
    std::sort(begin(g[qno]), begin(g[qno]) + top_k.num_rows());

    if (!std::equal(begin(top_k[qno]), begin(top_k[qno]) + k, begin(g[qno]))) {
      if (num_errors++ > 10) {
        return false;
      }
      std::cout << "Query " << qno << " is incorrect" << std::endl;
      for (size_t i = 0; i < std::min(k, static_cast<size_t>(10UL)); ++i) {
        std::cout << "  (" << top_k(i, qno) << " " << g(i, qno) << ")";
      }
      std::cout << std::endl;
    }
  }

  return true;
}


#ifdef TILEDB_VS_ENABLE_BLAS

/**
 * @brief Foreach input vector, apply a function to each element of the
 * vector and sum the resulting values
 * @tparam M
 * @tparam V
 * @param m
 * @param v
 * @param f
 * @return A vector containing the sum of the function applied down each column.
 */
template <class M, class V, class Function>
auto col_sum(
    const M& m, V& v, Function f = [](auto& x) -> const auto& { return x; }) {
  int size_m = size(m);
  int size_m0 = size(m[0]);

  for (int j = 0; j < size_m; ++j) {
    decltype(v[0]) vj = v[j];
    for (int i = 0; i < size_m0; ++i) {
      vj += f(m[j][i]);
    }
    v[j] = vj;
  }
}

/**
 * @brief Same as above, but for columns of a matrix rather than a collection
 * of vectors.
 */
template <class M, class V, class Function>
auto mat_col_sum(
    const M& m, V& v, Function f = [](auto& x) -> const auto& { return x; }) {
  auto num_cols = m.num_cols();
  auto num_rows = m.num_rows();

  for (size_t j = 0; j < num_cols; ++j) {
    decltype(v[0]) vj = v[j];
    for (size_t i = 0; i < num_rows; ++i) {
      vj += f(m(i, j));
    }
    v[j] = vj;
  }
}

/**
 * Query using dense linear algebra.  This uses the vector generalization of
 * the identity (a - b) * (a - b) = a * a + b * b - 2 * a * b .
 * We use outer products to compute the a * a and b * b terms, and then use
 * a gemm to compute the a * b term.
 *
 * This is extremely fast for large numbers of query vectors, but is not as fast
 * as vq_ew for small numbers of query vectors.
 */
template <class Matrix1, class Matrix2, class Matrix3>
void gemm_scores(const Matrix1& A, const Matrix2& B, Matrix3& C, unsigned nthreads) requires(
    (std::is_same_v<typename Matrix1::value_type, float> &&
     std::is_same_v<typename Matrix2::value_type, float> &&
     std::is_same_v<typename Matrix3::value_type, float>)) {
  using T = typename Matrix1::value_type;

  size_t M = A.num_cols();  // Vector dimension
  size_t N = B.num_cols();
  size_t K = A.num_rows();

  std::vector<T> alpha(M, 0.0f);
  std::vector<T> beta(N, 0.0f);
  std::vector<T> alpha_ones(N, 1.0f);
  std::vector<T> beta_ones(M, 1.0f);
  auto raveled_C = raveled(C);

  cblas_sgemm(
      CblasColMajor,
      CblasTrans,
      CblasNoTrans,
      M,
      N,
      K,
      -2.0,
      A.data(),
      K,
      B.data(),
      K,
      0.0,
      C.data(),
      M);

  mat_col_sum(
      A, alpha, [](auto a) { return a * a; });  // @todo optimize somehow
  mat_col_sum(B, beta, [](auto a) { return a * a; });

  cblas_sger(
      CblasColMajor, M, N, 1.0, &alpha[0], 1, &alpha_ones[0], 1, C.data(), M);
  cblas_sger(
      CblasColMajor, M, N, 1.0, &beta_ones[0], 1, &beta[0], 1, C.data(), M);

  stdx::execution::parallel_policy par{nthreads};
//  stdx::for_each(std::move(par), begin(raveled_C), end(raveled_C), [](auto& a) {
//    a = sqrt(a);
//  });
}

template <class Matrix1, class Matrix2, class Matrix3>
void gemm_scores(const Matrix1& A, const Matrix2& B, Matrix3& C, unsigned nthreads) requires(
    ((!std::is_same_v<typename Matrix1::value_type, float>)&&std::
         is_same_v<typename Matrix2::value_type, float> &&
     std::is_same_v<typename Matrix3::value_type, float>)) {
  ColMajorMatrix<float> A_f(A.num_rows(), A.num_cols());
  std::copy(A.data(), A.data() + A.num_rows() * A.num_cols(), A_f.data());

  gemm_scores(A_f, B, C, nthreads);
}

template <class Matrix1, class Matrix2, class Matrix3>
void gemm_scores(
    const Matrix1& A,
    const Matrix2& B,
    Matrix3& C,
    unsigned nthreads) requires(((!std::
                                      is_same_v<
                                          typename Matrix1::value_type,
                                          float>)&&(!std::
                                                        is_same_v<
                                                            typename Matrix2::
                                                                value_type,
                                                            float>)&&std::
                                     is_same_v<
                                         typename Matrix3::value_type,
                                         float>)) {
  ColMajorMatrix<float> A_f(A.num_rows(), A.num_cols());
  std::copy(A.data(), A.data() + A.num_rows() * A.num_cols(), A_f.data());

  ColMajorMatrix<float> B_f(B.num_rows(), B.num_cols());
  std::copy(B.data(), B.data() + B.num_rows() * B.num_cols(), B_f.data());

  gemm_scores(A_f, B_f, C, nthreads);
}

template <class Matrix1, class Matrix2>
auto gemm_scores(const Matrix1& A, const Matrix2& B, unsigned nthreads) {
  auto C = ColMajorMatrix<float>(A.num_cols(), B.num_cols());
  gemm_scores(A, B, C, nthreads);

  return C;
}
#endif // TILEDB_VS_ENABLE_BLAS
#endif  // TDB_SCORING_H