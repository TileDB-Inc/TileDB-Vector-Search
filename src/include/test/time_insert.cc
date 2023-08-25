/**
 * @file   time_insert.cc
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
 * Program to test performance of different min-heap implementations.
 *
 */

#include <numeric>
#include <random>
#include "utils/logging.h"

template <class Heap>
void do_time_heap_of_pair(
    const std::string& msg,
    Heap& heap,
    const std::vector<size_t>& v,
    size_t trials = 1) {
  scoped_timer _{msg};

  auto size_v = v.size();

  for (unsigned iter = 0; iter < trials; ++iter) {
    for (unsigned i = 0; i < size_v; ++i) {
      heap.insert({v[i], i});
    }
  }
}

template <class Heap>
void do_time_pair_heap(
    const std::string& msg,
    Heap& heap,
    const std::vector<size_t>& v,
    size_t trials = 1) {
  scoped_timer _{msg, true};

  auto size_v = v.size();

  for (unsigned iter = 0; iter < trials; ++iter) {
    for (unsigned i = 0; i < size_v; ++i) {
      heap.insert(v[i], i);
    }
  }
}

template <class T, class U, bool use_push = false, bool use_pop = false>
class simple_pair_heap : public std::vector<std::tuple<T, U>> {
  using Base = std::vector<std::tuple<T, U>>;
  // using Base::Base;
  unsigned max_size{0};

 public:
  explicit simple_pair_heap(std::integral auto k)
      : Base(0)
      , max_size{(unsigned)k} {
    Base::reserve(k);
  }

  explicit simple_pair_heap(
      unsigned k, std::initializer_list<std::tuple<T, U>> l)
      : Base(0)
      , max_size{k} {
    Base::reserve(k);
    for (auto& p : l) {
      insert(std::get<0>(p), std::get<1>(p));
    }
  }

  bool insert(const T& x, const U& y) {
    if (Base::size() < max_size) {
      Base::emplace_back(x, y);
      if constexpr (use_push) {
        std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
          return std::get<0>(a) < std::get<0>(b);
        });
      } else {
        if (Base::size() == max_size) {
          std::make_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
            return std::get<0>(a) < std::get<0>(b);
          });
        }
      }
      return true;
    } else if (x < std::get<0>(this->front())) {
      std::pop_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return std::get<0>(a) < std::get<0>(b);
      });

      if constexpr (use_pop) {
        this->pop_back();
        this->emplace_back(x, y);
      } else {
        // std::get<0>(this->back()) = x;
        // std::get<1>(this->back()) = y;
        (*this)[max_size - 1] = std::make_tuple(x, y);
        // std::get<0>((*this)[max_size - 1]) = x;
        // std::get<1>((*this)[max_size - 1]) = y;
      }

      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return std::get<0>(a) < std::get<0>(b);
      });
      return true;
    }
    return false;
  }
};

int main() {
  // Use a random device as the seed for the random number generator
  std::random_device rd;
  std::mt19937 rng(rd());

  for (size_t n : {10, 1000, 100'000, 100'000'000}) {
    std::vector<float> scores(n);
    std::vector<size_t> v(n);

    std::iota(begin(v), end(v), 17);
    std::iota(begin(scores), end(scores), 17);

    std::shuffle(begin(v), end(v), rng);

    for (unsigned k : {1, 10, 100}) {
      auto trials = (10 * 100'000'000UL) / (n) + 1;

      std::cout << n << " " << k << " " << trials << std::endl;

      auto heap_no_no = simple_pair_heap<float, size_t, false, false>(k);
      do_time_pair_heap("no push no pop", heap_no_no, v, trials);

      auto heap_no_yes = simple_pair_heap<float, size_t, false, true>(k);
      do_time_pair_heap("no push pop", heap_no_yes, v, trials);

      auto heap_yes_no = simple_pair_heap<float, size_t, true, false>(k);
      do_time_pair_heap("push no pop", heap_yes_no, v, trials);

      auto heap_yes_yes = simple_pair_heap<float, size_t, true, true>(k);
      do_time_pair_heap("push pop", heap_yes_yes, v, trials);

      std::cout << "--------------------------------------------------------\n";
    }
  }
}