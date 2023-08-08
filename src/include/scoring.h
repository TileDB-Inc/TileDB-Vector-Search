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
#include "defs.h"
#include "linalg.h"
#include "utils/timer.h"

#include "detail/linalg/choose_blas.h"


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
  using element = std::pair<float, unsigned>;
  fixed_min_heap<element> s(k);

  auto num_scores = scores.size();
  for (size_t i = 0; i < num_scores; ++i) {
    s.insert({scores[i], i});
  }
  std::sort_heap(begin(s), end(s));
  std::transform(
      s.begin(), s.end(), top_k.begin(), ([](auto&& e) { return e.second; }));

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