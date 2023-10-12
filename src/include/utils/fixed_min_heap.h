/**
 * @file   fixed_min_queues.h
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
 * Contains two implementations of a fixed-size min-heap (to experiment with
 * potential performance differences). Also contains an implementation of a
 * fixed_size min-heap for pairs.
 *
 * This type of heap is used to maintain the top k small scores as we compute
 * scores during similarity search.
 */

#ifndef TILEDB_FIXED_MIN_QUEUES_H
#define TILEDB_FIXED_MIN_QUEUES_H

#include <functional>
#include <initializer_list>
#include <set>

template <class T>
class fixed_min_set_heap_1 : public std::vector<T> {
  using Base = std::vector<T>;
  // using Base::Base;
  unsigned max_size{0};

 public:
  explicit fixed_min_set_heap_1(unsigned k)
      : Base(0)
      , max_size{k} {
    Base::reserve(k);
  }

  void insert(T const& x) {
    if (Base::size() < max_size) {
      this->push_back(x);
      std::push_heap(begin(*this), end(*this), std::less<T>());
    } else if (x < this->front()) {
      std::pop_heap(begin(*this), end(*this), std::less<T>());
      this->pop_back();
      this->push_back(x);
      std::push_heap(begin(*this), end(*this), std::less<T>());
    }
  }

  //  template<typename T, typename... Args>
  //  void my_emplace_back(std::vector<T>& vec, Args&&... args) {
  //    vec.emplace_back(std::forward<Args>(args)...);
  //  }
};

template <class T>
class fixed_min_set_heap_2 : public std::vector<T> {
  using Base = std::vector<T>;
  // using Base::Base;
  unsigned max_size{0};

 public:
  explicit fixed_min_set_heap_2(unsigned k)
      : Base(0)
      , max_size{k} {
    Base::reserve(k);
  }

  void insert(T const& x) {
    if (Base::size() < max_size) {
      this->push_back(x);
      std::push_heap(begin(*this), end(*this));
    } else if (x < this->front()) {
      // std::pop_heap(begin(*this), end(*this), std::less<T>());
      std::pop_heap(begin(*this), end(*this));
      this->pop_back();
      this->push_back(x);
      // std::push_heap(begin(*this), end(*this), std::less<T>());
      std::push_heap(begin(*this), end(*this));
    }
  }
};

/**
 * Heap to store a pair of values, ordered by the first element.
 * @tparam T Type of first element
 * @tparam U Type of second element
 */
template <class T, class U, class Compare = std::less<T>>
class fixed_min_pair_heap : public std::vector<std::tuple<T, U>> {
  using Base = std::vector<std::tuple<T, U>>;

  // using Base::Base;
  unsigned max_size{0};
  Compare compare_;

 public:
  explicit fixed_min_pair_heap(unsigned k, Compare compare = Compare())
      : Base(0)
      , max_size{k}
      , compare_{std::move(compare)} {
    Base::reserve(k);
  }

  explicit fixed_min_pair_heap(
      unsigned k,
      std::initializer_list<std::tuple<T, U>> l,
      Compare compare = Compare())
      : Base(0)
      , max_size{k}
      , compare_{std::move(compare)} {
    Base::reserve(k);
    for (auto& p : l) {
      insert(std::get<0>(p), std::get<1>(p));
    }
  }

  void insert(const T& x, const U& y) {
    if (Base::size() < max_size) {
      this->emplace_back(x, y);
      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });
    } else if (compare_(x, std::get<0>(this->front()))) {
      std::pop_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });
      this->pop_back();
      this->emplace_back(x, y);
      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });
    }
  }
};

template <class Heap>
struct heap_traits {
  using value_type = typename Heap::value_type;
  using score_type = typename std::tuple_element<0, typename Heap::value_type>::type;
  using index_type = typename std::tuple_element<1, typename Heap::value_type>::type;
};

template <class Heap>
using heap_score_t = typename heap_traits<Heap>::score_type;

template <class Heap>
using heap_index_t = typename heap_traits<Heap>::index_type;


// template <class T>
// using fixed_min_heap = fixed_min_set_heap_1<T>;

#ifdef ALLHEAPS  // Kept here for historical comparison reasons.  They are
                 // really slow.
template <class T, class Compare = std::less<T>>
class fixed_min_set_heap_3 : public std::vector<T> {
  using Base = std::vector<T>;
  // using Base::Base;
  unsigned max_size{0};
  Compare comp;

 public:
  explicit fixed_min_set_heap_3(unsigned k)
      : Base(0)
      , max_size{k} {
    Base::reserve(k);
  }
  fixed_min_set_heap_3(unsigned k, Compare c)
      : Base(0)
      , max_size{k}
      , comp{std::move(c)} {
    Base::reserve(k);
  }
  void insert(T const& x) {
    if (Base::size() < max_size) {
      Base::push_back(x);
      // std::push_heap(begin(*this), end(*this), std::less<T>());
      if (Base::size() == max_size) {
        std::make_heap(begin(*this), end(*this), comp);
      }
    } else if (comp(x, this->front())) {
      std::pop_heap(begin(*this), end(*this), comp);
      this->pop_back();
      this->push_back(x);
      std::push_heap(begin(*this), end(*this), comp);
    }
  }
};

template <
    class T,
    class Compare = std::less<T>,
    class Allocator = std::allocator<T>>
struct fixed_min_set_set : public std::set<T, Compare, Allocator> {
  using base = std::set<T, Compare, Allocator>;
  using base::base;

  unsigned max_size{0};

  explicit fixed_min_set_set(unsigned k)
      : max_size{k} {
  }
  fixed_min_set_set(unsigned k, const Compare& comp)
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
#endif
#endif  // TILEDB_FIXED_MIN_QUEUES_H
