//
// Created by Andrew Lumsdaine on 4/12/23.
//

#ifndef TDB_DEFS_H
#define TDB_DEFS_H

#include <algorithm>
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

  auto size_a = size(a);
  for (decltype(a.size()) i = 0; i < size_a; ++i) {
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

  auto size_a = size(a);
  for (auto i = 0; i < size_a; ++i) {
    sum += a[i] * b[i];
    a2 += a[i] * a[i];
    b2 += b[i] * b[i];
  }
  return sum / std::sqrt(a2 * b2);
}

template <class M, class V, class Function>
auto col_sum(const M& m, V& v, Function f = [](auto& x) -> const auto &{ return x; }) {
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


template <class V, class L, class I>
auto verify_top_k(V const& scores, L const& top_k, I const& g, int k, int qno) {
  if (!std::equal(begin(top_k), begin(top_k) +  k, g.begin(), [&](auto& a, auto& b) {
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

template <class L, class I>
auto verify_top_k(L const& top_k, I const& g, int k, int qno) {
  if (!std::equal(begin(top_k), begin(top_k) + k, g.begin())) {
    std::cout << "Query " << qno << " is incorrect" << std::endl;
    for (int i = 0; i < std::min(k, 10); ++i) {
      std::cout << "  (" << top_k[i] << " " << g[i] <<")";
    }
    std::cout << std::endl;
  }
}



template <class T, class Compare = std::less<T>, class Allocator = std::allocator<T>>
struct fixed_min_set : public std::set<T, Compare, Allocator> {
  using base = std::set<T, Compare, Allocator>;
  using base::base;

  unsigned max_size{0};

  explicit fixed_min_set(unsigned k) : max_size{k} { }
  fixed_min_set(unsigned k, const Compare& comp) : base(comp), max_size{k} {}

  bool at_max_size_ { false };
  T max_value_ = std::numeric_limits<T>::max();

  void insert(T const& x) {
    if (at_max_size_) {
      if (x < max_value_) {
	base::insert(x);
	base::erase(std::prev(base::end()));
	max_value_ = *(std::prev(base::end()));
      }
    } else {
      base::insert(x);
      max_value_ = *(std::prev(base::end()));
      if (max_size == base::size()) {
	at_max_size_ = true;
      }
    }
  }
};


template <class T, class U, class Compare = std::less<std::pair<T,U>>, class Allocator = std::allocator<std::pair<T,U>>>
struct fixed_min_set_pair : public std::set<std::pair<T, U>, Compare, Allocator> {
  using base = std::set<std::pair<T, U>, Compare, Allocator>;
  using base::base;

  unsigned max_size{0};

  explicit fixed_min_set_pair(unsigned k) : max_size{k} { }
  fixed_min_set_pair(unsigned k, const Compare& comp) : base(comp), max_size{k} {}

  bool at_max_size_ { false };
  std::pair<T, U> max_value_ {std::numeric_limits<T>::max(), std::numeric_limits<U>::max()};

  void insert(std::pair<T,U> const& x) {
    if (at_max_size_) {
      if (x.first < max_value_.first) {
	base::insert(x);
	base::erase(std::prev(base::end()));
	max_value_ = *(std::prev(base::end()));
      }
    } else {
      base::insert(x);
      max_value_ = *(std::prev(base::end()));
      if (max_size == base::size()) {
	at_max_size_ = true;
      }
    }
  }
};


template <class V, class L, class I>
auto get_top_k(V const& scores, L & top_k, I & index, int k) {
#if 1
  std::nth_element(begin(index), begin(index) + k, end(index), [&](auto&& a, auto&& b) {
    return scores[a] < scores[b];
  });
  std::copy(begin(index), begin(index) + k, begin(top_k));

  std::sort(begin(top_k), end(top_k), [&](auto& a, auto& b) {
    return scores[a] < scores[b];
  });
#else

  using element = std::pair<float, int>;
  auto min_k {fixed_min_set<element>(k)};

  auto scores_size = size(scores);

  for (size_t j = 0; j < scores_size; ++j) {
    min_k.insert(element{scores[j], index[j]});
  }
  // Copy indexes into top_k
  std::transform(min_k.begin(), min_k.end(), top_k.begin(), ([](auto &e) { return e.second; }));

  // Try to break ties by sorting the top k
  std::sort(begin(top_k), end(top_k));

#endif
}


template <class S, class T>
void get_top_k(const S& scores, T& top_k, int k, int size_q, int size_db, int nthreads) {
  life_timer _{"Get top k"};

  std::vector<int> i_index(size_db);
  std::iota(begin(i_index), end(i_index), 0);
  
  int q_block_size = (size_q + nthreads -1) / nthreads;
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);
  
  for (int n = 0; n < nthreads; ++n) {
    
    int q_start = n*q_block_size;
    int q_stop = std::min<int>((n+1)*q_block_size, size_q);
    
    futs.emplace_back(std::async(std::launch::async, [q_start, q_stop, &i_index, &scores, &top_k, k, size_db]() {
      
      std::vector<int> index(size_db);
      
      for (int j = q_start; j < q_stop; ++j) {
	// std::copy(begin(i_index), end(i_index), begin(index));
	std::iota(begin(index), end(index), 0);
	get_top_k(scores[j], top_k[j], index, k);
      }
    }));
  }      
  for (int n = 0; n < nthreads; ++n) {
    futs[n].get();
  }
}

#endif//TDB_DEFS_H
