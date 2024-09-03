/**
 * @file   fixed_min_heap.h
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
 *
 * @todo Run abstraction penalty benchmarks to make sure Distance is getting
 * completely inlined
 */

#ifndef TILEDB_FIXED_MIN_HEAP_H
#define TILEDB_FIXED_MIN_HEAP_H

#include <concepts>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <set>

#include "functional.h"

namespace {
class not_unique {};
class unique_id {};
class unique_score {};
class unique_both {};
}  // namespace

template <class Heap>
struct heap_traits {
  using value_type = typename Heap::value_type;
  using score_type =
      typename std::tuple_element<0, typename Heap::value_type>::type;
  using index_type =
      typename std::tuple_element<1, typename Heap::value_type>::type;
};

template <class Heap>
using heap_score_t = typename heap_traits<Heap>::score_type;

template <class Heap>
using heap_index_t = typename heap_traits<Heap>::index_type;

/**
 * Heap to store a tuple of values, ordered by the first element.
 * Supports pairs and triplets.
 * @tparam Tuple Type of tuple (pair or triplet)
 */
template <
    class Tuple,
    class Compare = std::less<typename std::tuple_element<0, Tuple>::type>>
class fixed_min_tuple_heap : public std::vector<Tuple> {
  using Base = std::vector<Tuple>;
  using T = typename std::tuple_element<0, Tuple>::type;
  using U = typename std::tuple_element<1, Tuple>::type;

  template <
      class TupleType,
      bool HasThirdElement = (std::tuple_size<TupleType>::value == 3)>
  struct ExtraTypeHelper {
    using type = typename std::tuple_element<2, TupleType>::type;
  };

  template <class TupleType>
  struct ExtraTypeHelper<TupleType, false> {
    using type = void*;
  };

  unsigned max_size{0};
  constexpr const static Compare compare_{};

 protected:
  using Extra = typename ExtraTypeHelper<Tuple>::type;

 public:
  explicit fixed_min_tuple_heap(
      std::integral auto k, Compare compare = Compare{})
      : Base(0)
      , max_size{(unsigned)k} {
    Base::reserve(k);
  }

  explicit fixed_min_tuple_heap(
      unsigned k, std::initializer_list<Tuple> l, Compare compare = Compare{})
      : Base(0)
      , max_size{k} {
    Base::reserve(k);
    for (const auto& p : l) {
      insert_impl(std::get<0>(p), std::get<1>(p), get_extra(p));
    }
  }

  auto pop() {
    if (this->empty()) {
      return;
    }
    std::pop_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
      return compare_(std::get<0>(a), std::get<0>(b));
    });
    this->pop_back();
  }

  void self_heapify() {
    std::make_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
      return compare_(std::get<0>(a), std::get<0>(b));
    });
  }

  void self_sort() {
    std::sort_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
      return compare_(std::get<0>(a), std::get<0>(b));
    });
  }

  std::string dump() const {
    std::ostringstream oss;
    if constexpr (std::tuple_size_v<Tuple> == 2) {
      for (const auto& [score, id] : *this) {
        oss << "(" << score << ", " << id << ") ";
      }
    } else {
      for (const auto& [score, id, extra] : *this) {
        oss << "(" << score << ", " << id << ", " << extra << ") ";
      }
    }
    return oss.str();
  }

 protected:
  template <class Unique = not_unique>
  bool insert_impl(const T& x, const U& y, const Extra& z = Extra{}) {
    if (max_size == 0) {
      return false;
    }
    if (Base::size() < max_size) {
      if constexpr (std::is_same_v<Unique, unique_id>) {
        if (std::find_if(begin(*this), end(*this), [y](auto&& e) {
              return std::get<1>(e) == y;
            }) != end(*this)) {
          return false;
        }
      }

      if constexpr (std::tuple_size_v<Tuple> == 2) {
        Base::emplace_back(x, y);
      } else {
        Base::emplace_back(x, y, z);
      }

      std::push_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });
      return true;
    } else if (compare_(x, std::get<0>(this->front()))) {
      std::pop_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });

      if constexpr (std::is_same_v<Unique, unique_id>) {
        if (std::find_if(begin(*this), end(*this), [y](auto&& e) {
              return std::get<1>(e) == y;
            }) != end(*this)) {
          std::push_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
            return compare_(std::get<0>(a), std::get<0>(b));
          });
          return false;
        }
      }

      if constexpr (std::tuple_size_v<Tuple> == 2) {
        (*this)[max_size - 1] = Tuple(x, y);
      } else {
        (*this)[max_size - 1] = Tuple(x, y, z);
      }

      std::push_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });
      return true;
    }
    return false;
  }

  // returns { inserted, evicted, evicted_score, evicted_id }
  // Cases:
  // 1. Inserted, not evicted: { true, false, x, y }
  // 2. Inserted, evicted: { true, true, old_score, old_id }
  // 3. Not inserted, not evicted: { false, false, x, y }
  // 4. Not inserted, evicted: exception
  template <class Unique = not_unique>
  std::tuple<bool, bool, T, U> evict_insert_impl(
      const T& x, const U& y, const Extra& z = Extra{}) {
    // There is room in the heap for the new element
    if (Base::size() < max_size) {
      // If the element id already exists in the heap, return false
      // We don't insert the element -- return inserted = false
      if constexpr (std::is_same_v<Unique, unique_id>) {
        if (std::find_if(begin(*this), end(*this), [y](auto&& e) {
              return std::get<1>(e) == y;
            }) != end(*this)) {
          // Not inserted
          return {false, false, x, y};
        }
      }

      // Insert, since there is room
      if constexpr (std::tuple_size_v<Tuple> == 2) {
        Base::emplace_back(x, y);
      } else {
        Base::emplace_back(x, y, z);
      }

      std::push_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });

      // Inserted, not evicted
      return {true, false, x, y};
    } else if (compare_(x, std::get<0>(this->front()))) {
      // If x < max_score in the heap, evict max_score and insert x
      // return inserted = true, evicted = true, old_score, old_id

      // Get the old element
      auto tmp = this->front();
      std::pop_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });

      if constexpr (std::is_same_v<Unique, unique_id>) {
        // If the new element id exists in the heap, return inserted = false
        if (std::find_if(begin(*this), end(*this), [y](auto&& e) {
              return std::get<1>(e) == y;
            }) != end(*this)) {
          // Since we had previously popped the heap, we need to unpop it
          std::push_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
            return compare_(std::get<0>(a), std::get<0>(b));
          });
          return {false, false, x, y};
        }
      }

      // Replace the former max element with the new element and re-heapify
      if constexpr (std::tuple_size_v<Tuple> == 2) {
        (*this)[max_size - 1] = Tuple(x, y);
      } else {
        (*this)[max_size - 1] = Tuple(x, y, z);
      }
      std::push_heap(begin(*this), end(*this), [&](const auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });

      // Inserted, evicted: return old element
      return {true, true, std::get<0>(tmp), std::get<1>(tmp)};
    }

    // If the new element is larger than the max, return not inserted
    return {false, false, x, y};
  }

 private:
  template <typename TupleType>
  static constexpr auto get_extra(const TupleType& t) {
    if constexpr (std::tuple_size_v<TupleType> == 3) {
      return std::get<2>(t);
    } else {
      return Extra{};
    }
  }
};

template <class T, class U, class Compare = std::less<T>>
class fixed_min_pair_heap
    : public fixed_min_tuple_heap<std::tuple<T, U>, Compare> {
  using Base = fixed_min_tuple_heap<std::tuple<T, U>, Compare>;

 public:
  explicit fixed_min_pair_heap(
      std::integral auto k, Compare compare = Compare{})
      : Base(k, compare) {
  }

  explicit fixed_min_pair_heap(
      unsigned k,
      std::initializer_list<std::tuple<T, U>> l,
      Compare compare = Compare{})
      : Base(k, l, compare) {
  }

  template <class Unique = not_unique>
  bool insert(const T& x, const U& y) {
    return Base::template insert_impl<Unique>(x, y);
  }

  template <class Unique = not_unique>
  auto evict_insert(const T& x, const U& y) {
    return Base::template evict_insert_impl<Unique>(x, y);
  }
};

template <class T, class U>
using k_min_heap = fixed_min_pair_heap<T, U>;

template <class T, class U, class V, class Compare = std::less<T>>
class fixed_min_triplet_heap
    : public fixed_min_tuple_heap<std::tuple<T, U, V>, Compare> {
  using Base = fixed_min_tuple_heap<std::tuple<T, U, V>, Compare>;

 public:
  explicit fixed_min_triplet_heap(
      std::integral auto k, Compare compare = Compare{})
      : Base(k, compare) {
  }

  explicit fixed_min_triplet_heap(
      unsigned k,
      std::initializer_list<std::tuple<T, U, V>> l,
      Compare compare = Compare{})
      : Base(k, l, compare) {
  }

  template <class Unique = not_unique>
  bool insert(const T& x, const U& y, const Base::Extra& z) {
    return Base::template insert_impl<Unique>(x, y, z);
  }

  template <class Unique = not_unique>
  auto evict_insert(const T& x, const U& y, const Base::Extra& z) {
    return Base::template evict_insert_impl<Unique>(x, y, z);
  }
};

template <class Heap>
void debug_min_heap(
    const Heap& heap, const std::string& msg = "", int which = 2) {
  std::cout << msg;

  if (which == 0) {
    for (auto&& [score, id] : heap) {
      std::cout << score << " ";
    }
    std::cout << std::endl;
  } else if (which == 1) {
    for (auto&& [score, id] : heap) {
      std::cout << id << " ";
    }
    std::cout << std::endl;
  } else {
    for (auto&& [score, id] : heap) {
      std::cout << "( " << score << ", " << id << " ) ";
    }
    std::cout << std::endl;
  }
}

template <class T, class Compare = std::greater<>>
void max_heapify(
    std::vector<T>& heap, int i, int heap_size, Compare comp = Compare()) {
  int largest = i;
  int left = 2 * i + 1;
  int right = 2 * i + 2;

  if (left < heap_size && comp(heap[left], heap[largest])) {
    largest = left;
  }

  if (right < heap_size && comp(heap[right], heap[largest])) {
    largest = right;
  }

  if (largest != i) {
    std::swap(heap[i], heap[largest]);
    max_heapify(heap, largest, heap_size);
  }
}

template <class T, class Compare = std::greater<>>
void convert_to_max_heap(std::vector<T>& heap, Compare comp = Compare()) {
  for (int i = heap.size() / 2 - 1; i >= 0; --i) {
    max_heapify(heap, i, heap.size(), comp);
  }
}

template <class T, class Compare = std::less<>>
void min_heapify(
    std::vector<T>& heap, int i, int heap_size, Compare comp = Compare()) {
  int smallest = i;
  int left = 2 * i + 1;
  int right = 2 * i + 2;

  if (left < heap_size && comp(heap[left], heap[smallest])) {
    smallest = left;
  }

  if (right < heap_size && comp(heap[right], heap[smallest])) {
    smallest = right;
  }

  if (smallest != i) {
    std::swap(heap[i], heap[smallest]);
    min_heapify(heap, smallest, heap_size);
  }
}

template <class T>
void convert_to_min_heap(std::vector<T>& heap) {
  for (int i = heap.size() / 2 - 1; i >= 0; --i) {
    min_heapify(heap, i, heap.size());
  }
}

#endif  // TILEDB_FIXED_MIN_HEAP_H
