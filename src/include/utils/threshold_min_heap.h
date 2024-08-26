/**
 * @file threshold_min_heap.h
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
 */

#ifndef TILEDB_THRESHOLD_MIN_HEAP_H
#define TILEDB_THRESHOLD_MIN_HEAP_H

#include <concepts>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <set>

#include "functional.h"

template <class T, class U>
class threshold_min_pair_heap : public std::vector<std::tuple<T, U>> {
 private:
  using element = std::tuple<T, U>;
  using Base = std::vector<element>;

  T threshold_{std::numeric_limits<T>::max()};

  void rebuild_heap() {
    std::vector<element> new_heap;
    new_heap.reserve(this->size());

    for (auto&& value : *this) {
      if (std::get<0>(value) < threshold_) {
        new_heap.emplace_back(value);
      }
    }
    std::swap(new_heap, *this);
    std::make_heap(this->begin(), this->end(), first_less<element>{});
  }

 public:
  threshold_min_pair_heap() = default;
  threshold_min_pair_heap(T threshold)
      : threshold_(threshold) {
  }

  void set_threshold(T new_threshold) {
    if (new_threshold < threshold_) {
      threshold_ = new_threshold;
      rebuild_heap();
    }
  }

  void insert(const T& t, const U& u) {
    if (t < threshold_) {
      this->emplace_back(t, u);
      std::push_heap(begin(*this), end(*this), first_less<element>{});
    }
  }

  void insert(element value) {
    if (std::get<0>(value) < threshold_) {
      this->push_back(value);
      std::push_heap(begin(*this), end(*this), first_less<element>{});
    }
  }

  auto get_min() {
    if (empty(*this)) {
      throw std::out_of_range("Heap is empty.");
    }
    return this->front();
  }

  void pop() {
    if (empty(*this)) {
      throw std::out_of_range("Heap is empty.");
    }
    std::pop_heap(begin(*this), end(*this), first_less<element>{});
    this->pop_back();
  }

  void unfiltered_heapify() {
    std::push_heap(begin(*this), end(*this), first_less<element>{});
  }

  void filtered_heapify() {
    rebuild_heap();
  }
};

#endif  // TILEDB_THRESHOLD_MIN_HEAP_H
