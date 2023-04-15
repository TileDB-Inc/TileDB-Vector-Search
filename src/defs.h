//
// Created by Andrew Lumsdaine on 4/12/23.
//

#ifndef TDB_DEFS_H
#define TDB_DEFS_H

#include <span>
#include <cmath>

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
auto col_sum(const M& m, V& v, Function f) {
  auto aa = m.size();
  auto bb = m[0].size();

  for (size_t j = 0; j < m.size(); ++j) {
    for (size_t i = 0; i < m[j].size(); ++i) {
      v[j] += f(m[j][i]);
    }
  }
}

#endif//TDB_DEFS_H
