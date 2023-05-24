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
*
*/

#ifndef TILEDB_FIXED_MIN_QUEUES_H
#define TILEDB_FIXED_MIN_QUEUES_H

#include <functional>
#include <set>


template <class T, class Compare = std::less<T>>
class fixed_min_set_heap_1 : public std::vector<T> {
  using Base = std::vector<T>;
  // using Base::Base;
  unsigned max_size { 0 };
  Compare  comp;

public:
  explicit fixed_min_set_heap_1(unsigned k) : Base(0), max_size { k } {
    Base::reserve(k);
  }
  fixed_min_set_heap_1(unsigned k, Compare c) : Base(0), max_size { k }, comp { std::move(c) } {
    Base::reserve(k);
  }

  void insert(T const& x) {
    if (Base::size() < max_size) {
      Base::push_back(x);
      // std::push_heap(begin(*this), end(*this), std::less<T>());
      if (Base::size() == max_size) {
        std::make_heap(begin(*this), end(*this), std::less<T>());
      }
    } else if (x < this->front()) {
      std::pop_heap(begin(*this), end(*this), std::less<T>());
      this->pop_back();
      this->push_back(x);
      std::push_heap(begin(*this), end(*this), std::less<T>());
    }
  }
};

template <class T, class Compare = std::less<T>>
class fixed_min_set_heap_2 : public std::vector<T> {
  using Base = std::vector<T>;
  // using Base::Base;
  unsigned max_size { 0 };
  Compare  comp;

public:
  explicit fixed_min_set_heap_2(unsigned k) : Base(0), max_size { k } {
    Base::reserve(k);
  }
  fixed_min_set_heap_2(unsigned k, Compare c) : Base(0), max_size { k }, comp { std::move(c) } {
    Base::reserve(k);
  }

  void insert(T const& x) {
    if (Base::size() < max_size) {
      Base::push_back(x);
      // std::push_heap(begin(*this), end(*this), std::less<T>());
      if (Base::size() == max_size) {
        std::make_heap(begin(*this), end(*this), std::less<T>());
      }
    } else if (x < this->front()) {
      std::pop_heap(begin(*this), end(*this), std::less<T>());
      this->pop_back();
      this->push_back(x);
      std::push_heap(begin(*this), end(*this), std::less<T>());
    }
  }
};

template <class T, class Compare = std::less<T>>
using fixed_min_heap = fixed_min_set_heap_1<T, Compare>;

#if 0 // These are really slow
template <class T, class Compare = std::less<T>>
class fixed_min_set_heap_3 : public std::vector<T> {
  using Base = std::vector<T>;
  // using Base::Base;
  unsigned max_size { 0 };
  Compare  comp;

public:
  explicit fixed_min_set_heap_3(unsigned k) : Base(0), max_size { k } {
    Base::reserve(k);
  }
  fixed_min_set_heap_3(unsigned k, Compare c) : Base(0), max_size { k }, comp { std::move(c) } {
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

template <class T, class Compare = std::less<T>, class Allocator = std::allocator<T>>
struct fixed_min_set_set : public std::set<T, Compare, Allocator> {
  using base = std::set<T, Compare, Allocator>;
  using base::base;

  unsigned max_size { 0 };

  explicit fixed_min_set_set(unsigned k) : max_size { k } {
  }
  fixed_min_set_set(unsigned k, const Compare& comp) : base(comp), max_size { k } {
  }

  bool maxed_ { false };

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
#endif // TILEDB_FIXED_MIN_QUEUES_H