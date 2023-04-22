//
// Created by Andrew Lumsdaine on 4/12/23.
//

#ifndef TDB_DEFS_H
#define TDB_DEFS_H

#include <span>
#include <cmath>
#include <future>
#include <iostream>
// #include <execution>
#include <memory>
#include <queue>
#include <set>

#include "timer.h"

template <class T>
using Vector = std::span<T>;

template <class V>
auto L2(V const& a, V const& b) {
  typename V::value_type sum {0};
  for (decltype(a.size()) i = 0; i < a.size(); ++i) {
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
  std::nth_element(begin(index), begin(index) + k, end(index), [&](auto a, auto b) {
    return scores[a] < scores[b];
  });
  std::copy(begin(index), begin(index) + k, begin(top_k));

  std::sort(begin(top_k), end(top_k), [&](auto a, auto b) {
    return scores[a] < scores[b];
  });
}

template <class V, class L, class I>
auto verify_top_k(V const& scores, L const& top_k, I const& g, size_t k, size_t qno) {
  if (!std::equal(begin(top_k), end(top_k), g.begin(), [&](auto a, auto b) {
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

template <class L, class I>
auto verify_top_k(L const& top_k, I const& g, size_t k, size_t qno) {
  if (!std::equal(begin(top_k), end(top_k), g.begin())) {
    std::cout << "Query " << qno << " is incorrect" << std::endl;
    for (size_t i = 0; i < 10; ++i) {
      std::cout << "  (" << top_k[i] << " " << g[i] <<")";
    }
    std::cout << std::endl;
  }
}



template <class T, class Compare = std::less<T>, class Allocator = std::allocator<T>>
struct fixed_min_set : public std::set<T, Compare, Allocator> {
  using base = std::set<T, Compare, Allocator>;
  using base::base;

  size_t max_size{0};

  explicit fixed_min_set(size_t k) : max_size{k} {}
  fixed_min_set(size_t k, const Compare& comp) : base(comp), max_size{k} {}

  void insert(T const& x) {
    base::insert(x);
    if (base::size() == max_size + 1) {
      base::erase(std::prev(base::end()));
    }
  }
};


template <class S, class T>
void get_top_k(const S& scores, T& top_k, size_t k, size_t size_q, size_t size_db, size_t nthreads) {
  life_timer _{"Get top k"};

  std::vector<int> i_index(size_db);
  std::iota(begin(i_index), end(i_index), 0);
  
  size_t q_block_size = (size_q + nthreads -1) / nthreads;
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);
  
  for (size_t n = 0; n < nthreads; ++n) {
    
    size_t q_start = n*q_block_size;
    size_t q_stop = std::min<size_t>((n+1)*q_block_size, size_q);
    
    futs.emplace_back(std::async(std::launch::async, [q_start, q_stop, &i_index, &scores, &top_k, k, size_db]() {
      
      std::vector<int> index(size_db);
      
      for (size_t j = q_start; j < q_stop; ++j) {
	std::copy(begin(i_index), end(i_index), begin(index));
	get_top_k(scores[j], top_k[j], index, k);
      }
    }));
  }      
  for (size_t n = 0; n < nthreads; ++n) {
    futs[n].get();
  }
}

#endif//TDB_DEFS_H
