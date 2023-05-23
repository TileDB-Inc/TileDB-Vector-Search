/**
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023-2023 TileDB, Inc.
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
 */

#ifndef TILEDB_VECTOR_SEARCH_VECTOR_DISTANCE_H
#define TILEDB_VECTOR_SEARCH_VECTOR_DISTANCE_H

#include <cmath>
#include <memory>
#include <iostream>
#include <queue>

namespace tiledb::vector_search {
/**
 * @brief Compute L2 distance between two vectors.
 * @tparam V
 * @param a
 * @param b
 * @return L2 norm of the difference between a and b.
 */
template <class V>
auto L2(V const& a, V const& b) {
  typename V::value_type sum{0};

  auto size_a = size(a);
  for (decltype(a.size()) i = 0; i < size_a; ++i) {
    auto diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}


auto L21(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  double sum = 0;
  for (int i = 0; i < a.size(); ++i) {
    double diff = int(a[i]) - int(b[i]);
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

auto L22(const std::vector<uint8_t>& a, const std::vector<float>& b) {
  double sum = 0;
  for (int i = 0; i < a.size(); ++i) {
    double diff = double(a[i]) - double(b[i]);
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

/**
 * A class for keeping a running tally of the top k elements in a set.
 * @tparam T
 * @tparam Compare
 * @tparam Allocator
 */
template <
    class T,
    class Compare = std::less<T>,
    class Allocator = std::allocator<T>>
struct fixed_min_set : public std::set<T, Compare, Allocator> {
  using base = std::set<T, Compare, Allocator>;
  using base::base;

  unsigned max_size{0};

  explicit fixed_min_set(unsigned k)
      : max_size{k} {
  }
  fixed_min_set(unsigned k, const Compare& comp)
      : base(comp)
      , max_size{k} {
  }

  bool maxed_{false};

  void insert(T const& x) {
    base::insert(x);
    if (maxed_) {
      base::erase(std::prev(base::end()));
    } else {
      if (base::size() == max_size) {
        maxed_ = true;
      }
    }
  }
};

template<
    class T,
    class Container = std::vector<T>,
    class Compare = std::less<typename Container::value_type>
>
class Topk {
  public:
    Topk(int k) : k_(k) {}
    void insert(T value) {
      if (q_.size() < k_) q_.push(value);
      else if (value < q_.top()) { 
         q_.pop(); q_.push(value); }
    }
    std::vector<int> finalize() {
      std::vector<int> result(q_.size());
      while (q_.size()) {
        result[q_.size() - 1] = q_.top();
        q_.pop();
      }
      return result;
    }
    int size(){return q_.size();}
    std::priority_queue<T> q_;
  private:
    int k_;
};

}  // namespace tiledb::vector_search

#endif  // TILEDB_VECTOR_SEARCH_VECTOR_DISTANCE_H