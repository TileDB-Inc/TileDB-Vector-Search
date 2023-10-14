/**
 * @file   flatpq_index.h
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
 * Header-only library of class that implements a flat index that used product
 * quantization.
 *
 */

#ifndef TILEDB_FLATPQ_INDEX_H
#define TILEDB_FLATPQ_INDEX_H

#include <cstddef>

#include "detail/flat/qv.h"

/**
 * @brief Run kmeans on a subspace
 * @tparam T Data type of the input vectors
 * @param sub_begin Beginning of the subspace
 * @param sub_end End of the subspace
 * @param num_clusters Number of clusters to find
 * @return
 *
 * @todo update with concepts
 */
template <class T>
auto sub_kmeans(
    ColMajorMatrix<T> training_set,
    size_t sub_begin,
    size_t sub_end,
    size_t num_clusters) {
}

/**
 * @brief Flat index that uses product quantization
 *
 * @tparam T Data type of the input vectors
 * @tparam shuffled_ids_type Data type of the shuffled ids
 * @tparam indices_type Data type of the indices
 */
template <
    class T,
    class shuffled_ids_type = size_t,
    class indices_type = size_t>
class flat_index {
  size_t dimension_{0};
  size_t num_subspaces_{0};
  size_t bits_per_subspace_{8};

 public:
  /**
   * @brief Construct a new flat index object
   * @param dimension Dimensionality of the input vectors
   * @param num_subspaces Number of subspaces (number of sections of the
   *       vector to quantize)
   * @param bits_per_subspace Number of bits per section (per subspace)
   *
   * @todo We don't really need dimension as an argument for any of our indexes
   */
  flat_index(
      size_t dimension, size_t num_subspaces, size_t bits_per_subspace = 8)
      : dimension_(dimension)
      , num_subspaces_(num_subspaces)
      , bits_per_subspace_(bits_per_subspace) {
    // Number of subspaces must evenly divide dimension of vector
    assert(dimension_ % num_subspaces_ == 0);
  }

  /**
   * @brief Train the index on a training set.  Run kmeans on each subspace and
   * create codewords from the centroids.
   *
   * @param training_set Training set
   */
  auto train(const ColMajorMatrix<T>& training_set) {
  }

  auto add() {
  }

  auto query() {
  }

  auto encode() {
  }

  auto decode() {
  }
};

#endif  // TILEDB_FLATPQ_INDEX_/**
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