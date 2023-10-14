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

template <
    class T,
    class shuffled_ids_type = size_t,
    class indices_type = size_t>
class flat_index {
  size_t dimension_{0};
  size_t num_subquantizers_{0};
  size_t bits_per_subquantizer_{0};

 public:
  /**
   * @brief Construct a new flat index object
   * @param dimension Dimensionality of the input vectors
   * @param num_subquantizers Number of subquantizers (number of sections of the
   *       vector to quantize)
   * @param bits_per_subquantizer Number of bits per section (per subvector)
   *
   * @todo We don't really need dimension as an argument for any of our indexes
   */
  flat_index(
      size_t dimension, size_t num_subquantizers, size_t bits_per_subquantizer)
      : dimension_(dimension)
      , num_subquantizers_(num_subquantizers)
      , bits_per_subquantizer_(bits_per_subquantizer) {
  }

  auto train() {
  }

  auto add() {
  }

  auto query() {
  }

  auto encode() {

  }

  auto decode () {

  }
};

#endif // TILEDB_FLATPQ_INDEX_H
