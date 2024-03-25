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
 * Heap to store a pair of values, ordered by the first element.
 * @tparam T Type of first element
 * @tparam U Type of second element
 */
template <class T, class U, class Compare = std::less<T>>
class fixed_min_pair_heap : public std::vector<std::tuple<T, U>> {
  using Base = std::vector<std::tuple<T, U>>;
  // using Base::Base;
  unsigned max_size{0};
  constexpr const static Compare compare_{};

 public:
  explicit fixed_min_pair_heap(
      std::integral auto k, Compare compare = Compare{})
      : Base(0)
      , max_size{(unsigned)k}  //    , compare_{std::move(compare)}
  {
    Base::reserve(k);
  }

  explicit fixed_min_pair_heap(
      unsigned k,
      std::initializer_list<std::tuple<T, U>> l,
      Compare compare = Compare{})
      : Base(0)
      , max_size{k} {
    Base::reserve(k);
    for (auto& p : l) {
      insert(std::get<0>(p), std::get<1>(p));
    }
  }

  template <class Unique = not_unique>
  bool insert(const T& x, const U& y) {
    if (Base::size() < max_size) {
      if constexpr (std::is_same_v<Unique, unique_id>) {
        if (std::find_if(begin(*this), end(*this), [y](auto&& e) {
              return std::get<1>(e) == y;
            }) != end(*this)) {
          return false;
        }
      }

      Base::emplace_back(x, y);

      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });
      //      if (Base::size() == max_size) {
      //        std::make_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
      //          return std::get<0>(a) < std::get<0>(b);
      //        });
      //      }
      return true;
    } else if (compare_(x, std::get<0>(this->front()))) {
      std::pop_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });

      if constexpr (std::is_same_v<Unique, unique_id>) {
        if (std::find_if(begin(*this), end(*this), [y](auto&& e) {
              return std::get<1>(e) == y;
            }) != end(*this)) {
          std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
            return compare_(std::get<0>(a), std::get<0>(b));
          });
          return false;
        }
      }

      //      this->pop_back();
      //      this->emplace_back(x, y);
      (*this)[max_size - 1] = std::make_tuple(x, y);
      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
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
  std::tuple<bool, bool, T, U> evict_insert(const T& x, const U& y) {
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
      Base::emplace_back(x, y);
      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });

      // Inserted, not evicted
      return {true, false, x, y};
    } else if (compare_(x, std::get<0>(this->front()))) {
      // If x < max_score in the heap, evict max_score and insert x
      // return inserted = true, evicted = true, old_score, old_id

      // Get the old element
      auto tmp = this->front();
      std::pop_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });

      if constexpr (std::is_same_v<Unique, unique_id>) {
        // If the new element id exists in the heap, return inserted = false
        if (std::find_if(begin(*this), end(*this), [y](auto&& e) {
              return std::get<1>(e) == y;
            }) != end(*this)) {
          // Since we had previously popped the heap, we need to unpop it
          std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
            return compare_(std::get<0>(a), std::get<0>(b));
          });
          return {false, false, x, y};
        }
      }

      // Replace the former max element with the new element and re-heapify
      (*this)[max_size - 1] = std::make_tuple(x, y);
      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return compare_(std::get<0>(a), std::get<0>(b));
      });

      // Inserted, evicted: return old element
      return {true, true, std::get<0>(tmp), std::get<1>(tmp)};
    }

    // If the new element is larger than the max, return not inserted
    return {false, false, x, y};
  }

  auto pop() {
    std::pop_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
      return compare_(std::get<0>(a), std::get<0>(b));
    });
    this->pop_back();
  }

  void self_heapify() {
    std::make_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
      return compare(std::get<0>(a), std::get<0>(b));
    });
  }

  void self_sort() {
    std::sort_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
      return compare(std::get<0>(a), std::get<0>(b));
    });
  }
};

template <class T, class U>
using k_min_heap = fixed_min_pair_heap<T, U>;

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

template <class T, class U>
using threshold_heap = threshold_min_pair_heap<T, U>;

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

#ifdef ALLHEAPS

// Original simple heap
#if 0
/**
 * Heap to store a pair of values, ordered by the first element.
 * @tparam T Type of first element
 * @tparam U Type of second element
 */
template <class T, class U>
class fixed_min_pair_heap : public std::vector<std::tuple<T, U>> {
  using Base = std::vector<std::tuple<T, U>>;

  // using Base::Base;
  unsigned max_size{0};

 public:
  explicit fixed_min_pair_heap(unsigned k)
      : Base(0)
      , max_size{k} {
    Base::reserve(k);
  }

  explicit fixed_min_pair_heap(
      unsigned k, std::initializer_list<std::tuple<T, U>> l)
      : Base(0)
      , max_size{k} {
    Base::reserve(k);
    for (auto& p : l) {
      insert(std::get<0>(p), std::get<1>(p));
    }
  }

  void insert(const T& x, const U& y) {
    if (Base::size() < max_size) {
      this->emplace_back(x, y);
      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return std::get<0>(a) < std::get<0>(b);
      });
    } else if (x < std::get<0>(this->front())) {
      std::pop_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return std::get<0>(a) < std::get<0>(b);
      });
      this->pop_back();
      this->emplace_back(x, y);
      std::push_heap(begin(*this), end(*this), [&](auto& a, auto& b) {
        return std::get<0>(a) < std::get<0>(b);
      });
    }
  }
};
#endif

// Kept here for historical comparison reasons.
// These are okay

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
  using score_type =
      typename std::tuple_element<0, typename Heap::value_type>::type;
  using index_type =
      typename std::tuple_element<1, typename Heap::value_type>::type;
};

template <class Heap>
using heap_score_t = typename heap_traits<Heap>::score_type;

template <class Heap>
using heap_index_t = typename heap_traits<Heap>::index_type;

// template <class T>
// using fixed_min_heap = fixed_min_set_heap_1<T>;

// Kept here for historical comparison reasons.  They are
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

#endif  // ALLHEAPS
#endif  // TILEDB_FIXED_MIN_HEAP_H
