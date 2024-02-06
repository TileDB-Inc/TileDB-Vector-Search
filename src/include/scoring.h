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
#include "utils/utils.h"

#include "detail/linalg/choose_blas.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <ranges>
#include <set>
#include <span>
// #include <execution>

#include "detail/linalg/linalg_defs.h"
#include "tdb_defs.h"
#include "utils/fixed_min_heap.h"
#include "utils/timer.h"

#include "utils/print_types.h"

// ----------------------------------------------------------------------------
// Helper utilities
//----------------------------------------------------------------------------
namespace {
class with_ids {};
class without_ids {};
}  // namespace

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

  if constexpr (
      std::unsigned_integral<std::remove_reference_t<decltype(a[0])>> ||
      std::unsigned_integral<std::remove_reference_t<decltype(b[0])>>) {
    for (size_t i = 0; i < size_a; ++i) {
      // float diff = (float)a[i] - (float)b[i];  // converting to float is slow
      float diff = (float)a[i] - (float)b[i];
      sum += diff * diff;
    }
  } else {
    for (size_t i = 0; i < size_a; ++i) {
      // float diff = (float)a[i] - (float)b[i];  // converting to float is slow
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
  }
  return sum;
}

template <class V>
inline auto sum_of_squares(V const& a) {
  float sum{0.0};
  size_t size_a = size(a);

  if constexpr (std::unsigned_integral<
                    std::remove_reference_t<decltype(a[0])>>) {
    for (size_t i = 0; i < size_a; ++i) {
      // float diff = (float)a[i] - (float)b[i];  // converting to float is slow
      float diff = (float)a[i];
      sum += diff * diff;
    }
  } else {
    for (size_t i = 0; i < size_a; ++i) {
      // float diff = (float)a[i] - (float)b[i];  // converting to float is slow
      float diff = a[i];
      sum += diff * diff;
    }
  }
  return sum;
}

template <class V, class U>
inline auto sub_sum_of_squares(
    V const& a, U const& b, size_t start, size_t end) {
  float sum{0.0};

  if constexpr (
      std::unsigned_integral<std::remove_reference_t<decltype(a[0])>> ||
      std::unsigned_integral<std::remove_reference_t<decltype(b[0])>>) {
    for (size_t i = start; i < end; ++i) {
      // float diff = (float)a[i] - (float)b[i];  // converting to float is slow
      float diff = (float)a[i] - (float)b[i];
      sum += diff * diff;
    }
  } else {
    for (size_t i = start; i < end; ++i) {
      // float diff = (float)a[i] - (float)b[i];  // converting to float is slow
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
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
// Function objects for computing distances
// ----------------------------------------------------------------------------

struct sum_of_squares_distance {
  template <class V, class U>
  constexpr auto operator()(const V& a, const U& b) const {
    return sum_of_squares(a, b);
  }

  template <class V>
  constexpr auto operator()(const V& a) const {
    return sum_of_squares(a);
  }
};

using l2_distance = sum_of_squares_distance;
using L2_distance = sum_of_squares_distance;

struct sub_sum_of_squares_distance {
 private:
  size_t start_{0};
  size_t stop_{0};

 public:
  sub_sum_of_squares_distance(size_t start, size_t stop)
      : start_(start)
      , stop_(stop) {
  }
  template <class V, class U>
  constexpr auto operator()(const V& a, const U& b) const {
    return sub_sum_of_squares(a, b, start_, stop_);
  }
};

using sub_l2_distance = sub_sum_of_squares_distance;
using sub_L2_distance = sub_sum_of_squares_distance;

// ----------------------------------------------------------------------------
// Functions for dealing with the case of when size of scores < k_nn
// ----------------------------------------------------------------------------

// One-dimensional case
template <class U>
void pad_with_sentinels(size_t start, U& top_k) {
  using index_type = typename U::value_type;
  for (size_t i = start; i < top_k.size(); ++i) {
    top_k[i] = std::numeric_limits<index_type>::max();
  }
}

// One-dimensional case
template <class U, class V>
void pad_with_sentinels(size_t start, U& top_k, V& top_k_scores) {
  using score_type = typename V::value_type;
  using index_type = typename U::value_type;
  for (size_t i = start; i < top_k.size(); ++i) {
    top_k[i] = std::numeric_limits<index_type>::max();
    top_k_scores[i] = std::numeric_limits<score_type>::max();
  }
}

// One-dimensional case
template <class U>
void trim_top_k(size_t start, U& top_k) {
  top_k.resize(start);
}

// One-dimensional case
template <class U, class V>
void trim_top_k(size_t start, U& top_k, V& top_k_scores) {
  top_k.resize(start);
  top_k_scores.resize(start);
}

// ----------------------------------------------------------------------------
// Functions for extracting top k neighbors from a raw scores matrix
// ----------------------------------------------------------------------------

/**
 * @brief Get top k neighbors for each query. Scans the scores for each
 * @tparam V
 * @tparam L
 * @param scores
 * @param top_k
 * @param k
 * @return
 */

template <
    std::ranges::random_access_range V,
    std::ranges::random_access_range L>
auto get_top_k_from_scores(V const& scores, L&& top_k, size_t k = 0) {
  using value_type = typename V::value_type;
  using index_type = typename std::remove_reference_t<L>::value_type;

  auto num_scores = size(scores);

  if (k == 0) {
    k = num_scores;
  }

  fixed_min_pair_heap<value_type, index_type> s(k);

  for (size_t i = 0; i < num_scores; ++i) {
    s.insert(scores[i], i);
  }
  get_top_k_from_heap(s, top_k);
}

// Note that we cannot pad top_k with sentinels here because we don't know the
// size of the valid ranges in the scores matrix.
// @todo pad top_k with sentinel if scores has sentinel
template <class I, class T>
auto get_top_k_from_scores(const ColMajorMatrix<T>& scores, int k_nn) {
  auto top_k = ColMajorMatrix<I>(k_nn, scores.num_cols());
  for (size_t j = 0; j < scores.num_cols(); ++j) {
    get_top_k_from_scores(scores[j], top_k[j], k_nn);
  }
  return top_k;
}

// ----------------------------------------------------------------------------
// Functions for consolidating vector of vectors of min_heaps to 0th min_heap
// ----------------------------------------------------------------------------
/**
 * @brief Utility function to put the top scores for multiple threads into a
 * single top_scores vector (the zeroth vector).
 * @tparam Heap
 *   @param min_scores a vector of vectors of min_heaps.  Each vector of
 * min_heaps is the top k scores for a set of queries.  Each vector of vectors
 * is stores a vector of min_heaps, one per thread.
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

// ----------------------------------------------------------------------------
// Functions for extracting top k neighbor indices from a min heap (with pairs)
// ----------------------------------------------------------------------------
/**
 * @brief Utility function to extract the top k scores from a single min heap.
 * @param min_scores
 * @param top_k
 */
template <class Heap>
inline void get_top_k_from_heap(Heap& min_scores, auto&& top_k)
  requires(!std::is_same_v<Heap, std::vector<Heap>>)
{
  std::sort_heap(begin(min_scores), end(min_scores), [](auto&& a, auto&& b) {
    return std::get<0>(a) < std::get<0>(b);
  });
  auto k_nn = std::min(size(min_scores), size(top_k));
  std::transform(
      begin(min_scores), begin(min_scores) + k_nn, begin(top_k), ([](auto&& e) {
        return std::get<1>(e);
      }));
  pad_with_sentinels(k_nn, top_k);
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
template <class Heap>
inline auto get_top_k(std::vector<Heap>& scores, size_t k_nn) {
  // using score_type = heap_score_t<Heap>;
  using index_type = heap_index_t<Heap>;

  auto num_queries = size(scores);

  ColMajorMatrix<index_type> top_k(k_nn, num_queries);

  for (size_t j = 0; j < num_queries; ++j) {
    get_top_k_from_heap(scores[j], top_k[j]);  // Will pad with sentinels
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
template <class Heap>
inline auto get_top_k(std::vector<std::vector<Heap>>& scores, size_t k_nn) {
  return get_top_k(scores[0], k_nn);
}

// ----------------------------------------------------------------------------
// Functions for computing top k neighbors with scores
// ----------------------------------------------------------------------------

inline void get_top_k_with_scores_from_heap(
    auto&& min_scores, auto&& top_k, auto&& top_k_scores) {
  std::sort_heap(begin(min_scores), end(min_scores), [](auto&& a, auto&& b) {
    return std::get<0>(a) < std::get<0>(b);
  });
  auto k_nn = std::min(size(min_scores), size(top_k));
  std::transform(
      begin(min_scores),
      begin(min_scores) + k_nn,
      begin(top_k_scores),
      ([](auto&& e) { return std::get<0>(e); }));
  std::transform(
      begin(min_scores), begin(min_scores) + k_nn, begin(top_k), ([](auto&& e) {
        return std::get<1>(e);
      }));
  pad_with_sentinels(k_nn, top_k, top_k_scores);
}

template <class Heap>
inline void get_top_k_with_scores_from_heap(const Heap& min_scores, size_t k) {
  using element_type = std::remove_cvref_t<decltype(*(
      min_scores.begin()))>; /*typename Heap::value_type;*/
  using value_type = typename std::tuple_element<0, element_type>::type;
  using index_type = typename std::tuple_element<1, element_type>::type;

  auto top_k = Vector<index_type>(k);
  auto top_k_scores = Vector<value_type>(k);

  get_top_k_with_scores_from_heap(min_scores, top_k, top_k_scores);
  return std::make_tuple(std::move(top_k_scores), std::move(top_k));
}

// Overload for one-d scores
template <class Heap>
inline auto get_top_k_with_scores(std::vector<Heap>& scores, size_t k_nn) {
  using score_type = heap_score_t<Heap>;
  using index_type = heap_index_t<Heap>;

  auto num_queries = size(scores);

  ColMajorMatrix<index_type> top_k(k_nn, num_queries);
  ColMajorMatrix<score_type> top_scores(k_nn, num_queries);

  for (size_t j = 0; j < num_queries; ++j) {
    get_top_k_with_scores_from_heap(
        scores[j], top_k[j], top_scores[j]);  // Will pad with sentinels
  }
  return std::make_tuple(std::move(top_scores), std::move(top_k));
}

// Overload for two-d scores
template <class Heap>
inline auto get_top_k_with_scores(
    std::vector<std::vector<Heap>>& scores, size_t k_nn) {
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
 * Used only for testing / benchmarking where sentinels have not been needed, so
 * we don't need to handle the case where top_k might have sentinels.
 * @todo Handle the error more systematically and succinctly.
 */
template <class V, class L, class I>
auto verify_top_k_scores(
    V const& scores, L const& top_k, I const& g, int k, int qno) {
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

#if 0
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
#endif

/**
 * @brief Check the computed top k vectors against the ground truth.
 * This version is for approximate search and so will sort the results before
 * comparing.
 */
template <feature_vector_array TK, feature_vector_array G>
bool validate_top_k(TK& top_k, const G& g) {
  size_t k = dimension(top_k);
  size_t num_errors = 0;

  for (size_t qno = 0; qno < num_vectors(top_k); ++qno) {
    // @todo -- count intersections rather than testing for equality
    std::sort(begin(top_k[qno]), end(top_k[qno]));
    std::sort(begin(g[qno]), begin(g[qno]) + dimension(top_k));

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

template <feature_vector_array U, feature_vector_array V>
auto count_intersections(const U& I, const V& groundtruth, size_t k_nn) {
  // print_types(I, groundtruth);

  size_t total_intersected = 0;

  if constexpr (feature_vector_array<std::remove_cvref_t<decltype(I)>>) {
    for (size_t i = 0; i < I.num_cols(); ++i) {
      std::sort(begin(I[i]), end(I[i]));
      std::sort(begin(groundtruth[i]), begin(groundtruth[i]) + k_nn);

      // @todo remove -- for debugging only
      std::vector<size_t> x(begin(I[i]), end(I[i]));
      std::vector<size_t> y(begin(groundtruth[i]), end(groundtruth[i]));

      total_intersected += std::set_intersection(
          begin(I[i]),
          end(I[i]),
          begin(groundtruth[i]),
          /*end(groundtruth[i]*/ begin(groundtruth[i]) + k_nn,
          assignment_counter{});
    }
  } else {
    if constexpr (feature_vector<std::remove_cvref_t<decltype(I)>>) {
      std::sort(begin(I), end(I));
      std::sort(begin(groundtruth), begin(groundtruth) + k_nn);

      total_intersected += std::set_intersection(
          begin(I),
          end(I),
          begin(groundtruth),
          /*end(groundtruth)*/ begin(groundtruth) + k_nn,
          assignment_counter{});
    } else {
      static_assert(
          always_false<std::remove_cvref_t<decltype(I)>>,
          "T must be a feature_vector or feature_vector_array");
    }
  }
  return total_intersected;
};

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
void gemm_scores(
    const Matrix1& A, const Matrix2& B, Matrix3& C, unsigned nthreads)
  requires(
      (std::is_same_v<typename Matrix1::value_type, float> &&
       std::is_same_v<typename Matrix2::value_type, float> &&
       std::is_same_v<typename Matrix3::value_type, float>))
{
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
  //  stdx::for_each(std::move(par), begin(raveled_C), end(raveled_C), [](auto&
  //  a) {
  //    a = sqrt(a);
  //  });
}

template <class Matrix1, class Matrix2, class Matrix3>
void gemm_scores(
    const Matrix1& A, const Matrix2& B, Matrix3& C, unsigned nthreads)
  requires(
      ((!std::is_same_v<typename Matrix1::value_type, float>) &&
       std::is_same_v<typename Matrix2::value_type, float> &&
       std::is_same_v<typename Matrix3::value_type, float>))
{
  ColMajorMatrix<float> A_f(A.num_rows(), A.num_cols());
  std::copy(A.data(), A.data() + A.num_rows() * A.num_cols(), A_f.data());

  gemm_scores(A_f, B, C, nthreads);
}

template <class Matrix1, class Matrix2, class Matrix3>
void gemm_scores(
    const Matrix1& A, const Matrix2& B, Matrix3& C, unsigned nthreads)
  requires(
      ((!std::is_same_v<typename Matrix1::value_type, float>) &&
       (!std::is_same_v<typename Matrix2::value_type, float>) &&
       std::is_same_v<typename Matrix3::value_type, float>))
{
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
#endif  // TILEDB_VS_ENABLE_BLAS
#endif  // TDB_SCORING_H
