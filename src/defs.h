//
// Created by Andrew Lumsdaine on 4/12/23.
//

#ifndef TDB_DEFS_H
#define TDB_DEFS_H

#include <span>
#include <cmath>
#include <execution>

template <class T>
using Vector = std::span<T>;

template <class V>
auto L2(V const& a, V const& b) {
  typename V::value_type sum {0};
  for (auto i = 0; i < a.size(); ++i) {
    auto diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

template <class V>
auto cosine(V const& a, V const& b) {
  typename V::value_type sum {0};
  auto a2  = 0.0;
  auto b2  = 0.0;
  for (auto i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
    a2 += a[i] * a[i];
    b2 += b[i] * b[i];
  }
  return sum / std::sqrt(a2 * b2);
}

template <class M, class V, class Function>
auto col_sum(const M& m, V& v, Function f = [](auto x) { return x; }) {
  auto aa = size(m);
  auto bb = size(m[0]);

  for (size_t j = 0; j < size(m); ++j) {
    for (size_t i = 0; i < size(m[j]); ++i) {
      v[j] += f(m[j][i]);
    }
  }
}

template <class V, class L, class I>
auto get_top_k(V const& scores, L & top_k, I & index, size_t k) {
  std::nth_element(std::execution::seq, begin(index), begin(index) + k, end(index), [&](auto a, auto b) {
    return scores[a] < scores[b];
  });
  std::copy(std::execution::seq, begin(index), begin(index) + k, begin(top_k));

  std::sort(std::execution::seq, begin(top_k), end(top_k), [&](auto a, auto b) {
    return scores[a] < scores[b];
  });
}

template <class V, class L, class I>
auto verify_top_k(V const& scores, L const& top_k, I const& g, size_t k, size_t qno) {
  if (!std::equal(std::execution::par_unseq, begin(top_k), end(top_k), g.begin(), [&](auto a, auto b) {
    return scores[a] == scores[b];
  })) {
    std::cout << "Query " << qno << " is incorrect" << std::endl;
    for (size_t i = 0; i < k; ++i) {
      std::cout << "  (" << top_k[i] << " " << scores[top_k[i]] << ") ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < k; ++i) {
      std::cout << "  (" << g[i] << " " << scores[g[i]] << ") ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
}


#endif//TDB_DEFS_H
