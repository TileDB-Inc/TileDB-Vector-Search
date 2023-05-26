/**
 * @file   defs.h
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
 * Support functions for TiledB vector search algorithms.
 *
 */

#ifndef TDB_DEFS_H
#define TDB_DEFS_H

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

#include "fixed_min_queues.h"
#include "linalg.h"
#include "utils/timer.h"

/**
 * @brief Compute L2 distance between two vectors.
 * @tparam V
 * @param a
 * @param b
 * @return L2 norm of the difference between a and b.
 */
template <class V>
auto L2(V const& a, V const& b) {
  typename V::value_type sum { 0 };

  auto size_a = size(a);
  for (decltype(a.size()) i = 0; i < size_a; ++i) {
    auto diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
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
  typename V::value_type sum { 0 };
  auto                   a2 = 0.0;
  auto                   b2 = 0.0;

  auto size_a = size(a);
  for (auto i = 0; i < size_a; ++i) {
    sum += a[i] * b[i];
    a2 += a[i] * a[i];
    b2 += b[i] * b[i];
  }
  return sum / std::sqrt(a2 * b2);
}

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
  int size_m  = size(m);
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

  for (int j = 0; j < num_cols; ++j) {
    decltype(v[0]) vj = v[j];
    for (int i = 0; i < num_rows; ++i) {
      vj += f(m(i, j));
    }
    v[j] = vj;
  }
}


template <class L, class I>
auto verify_top_k_index(L const& top_k, I const& g, int k, int qno) {

  std::sort(begin(g), begin(g) + k);
  std::sort(begin(top_k), end(top_k));

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
  if (!std::equal(begin(top_k), begin(top_k) + k, g.begin(), [&](auto& a, auto& b) { return scores[a] == scores[b]; })) {
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

// @todo implement with fixed_min_set
template <class V, class L, class I>
auto get_top_k(V const& scores, L&& top_k, I& index, int k, bool nth = false) {
  if (nth) {
    std::nth_element(begin(index), begin(index) + k, end(index), [&](auto&& a, auto&& b) { return scores[a] < scores[b]; });
    std::copy(begin(index), begin(index) + k, begin(top_k));
    std::sort(begin(top_k), end(top_k), [&](auto& a, auto& b) { return scores[a] < scores[b]; });
  } else {
    using element = std::pair<float, unsigned>;
    fixed_min_heap<element> s(k);
    for (size_t i = 0; i < index.size(); ++i) {
      s.insert({ scores[index[i]], index[i] });
    }
    std::sort_heap(begin(s), end(s));
    std::transform(s.begin(), s.end(), top_k.begin(), ([](auto&& e) { return e.second; }));
  }
}

template <class S>
auto get_top_k(const S& scores, int k, bool nth, int nthreads) {
  life_timer _ { "Get top k" };

  auto num_queries = scores.num_cols();

  auto top_k = ColMajorMatrix<size_t>(k, num_queries);

  int                            q_block_size = (num_queries + nthreads - 1) / nthreads;
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (int n = 0; n < nthreads; ++n) {
    int q_start = n * q_block_size;
    int q_stop  = std::min<int>((n + 1) * q_block_size, num_queries);

    futs.emplace_back(std::async(std::launch::async, [q_start, q_stop, &scores, &top_k, k, nth]() {
      std::vector<int> index(scores.num_rows());

      for (int j = q_start; j < q_stop; ++j) {
        std::iota(begin(index), end(index), 0);
        get_top_k(scores[j], std::move(top_k[j]), index, k, nth);
      }
    }));
  }
  for (int n = 0; n < nthreads; ++n) {
    futs[n].get();
  }
  return top_k;
}

template <class TK, class G>
bool validate_top_k(TK& top_k, G& g) {

  size_t k          = top_k.num_rows();
  size_t num_errors = 0;
  for (size_t qno = 0; qno < top_k.num_cols(); ++qno) {
    std::sort(begin(top_k[qno]), end(top_k[qno]));
    std::sort(begin(g[qno]), begin(g[qno]) + top_k.num_rows());

    if (!std::equal(begin(top_k[qno]), begin(top_k[qno]) + k, begin(g[qno]))) {
      if (num_errors++ > 10) {
        return false;
      }
      std::cout << "Query " << qno << " is incorrect" << std::endl;
      for (int i = 0; i < std::min(k, 10UL); ++i) {
        std::cout << "  (" << top_k(i, qno) << " " << g(i, qno) << ")";
      }
      std::cout << std::endl;
    }
  }

  return true;
}

#endif    // TDB_DEFS_H
