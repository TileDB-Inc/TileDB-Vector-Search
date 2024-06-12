/**
 * @file   l2_distance.h
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
 * C++ functions for computing L2 distance between two feature vectors
 *
 * This file contains a number of different implementations of the L2 distance
 * computation, including naive, unrolled, and start/stop versions. The
 * implementations are templated on the type of the feature vector, and
 * concepts are used to select between feature vectors of float and uint8_t or
 * int8_t.
 *
 * Implementations include:
 * - naive_sum_of_squares: single loop, one statement in inner loop
 * - unroll4_sum_of_squares: 4x unrolled
 * - naive_sum_of_squares with start and stop
 * - unroll4_sum_of_squares with start and stop
 *
 * The unrolled versions use simple 4x unrolling, and are faster than the
 * naive versions. The start/stop versions are useful for computing the
 * "sub" distance, that is, the distance between just a portion of two vectors,
 * which is used in pq distance computation.
 *
 *
 * @todo Relax condition of uint8_t to integral types
 */
#ifndef TDB_SCORING_L2_DISTANCE_H
#define TDB_SCORING_L2_DISTANCE_H

#include "concepts.h"

/****************************************************************
 *
 * Naive algorithms (single loop, one statement in inner loop)
 *
 ****************************************************************/

/**
 * Compute l2 distance between two vectors of float
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&

           std::same_as<typename W::value_type, float>
inline float naive_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

/**
 * Compute l2 distance between vector of float and vector of uint8_t
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, uint8_t>
inline float naive_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = a[i] - (float)b[i];
    sum += diff * diff;
  }
  return sum;
}

/**
 * Compute l2 distance between vector of uint8_t and vector of float
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, float>
inline float naive_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = (float)a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

/**
 * Compute l2 distance between vector of uint8_t and vector of uint8_t
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, uint8_t>
inline float naive_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = (float)a[i] - (float)b[i];
    sum += diff * diff;
  }
  return sum;
}

/****************************************************************
 *
 * 4x unrolled algorithms
 *
 ****************************************************************/

/**
 * Unrolled l2 of single vector
 */
template <feature_vector V>
  requires std::same_as<typename V::value_type, float> ||
           std::same_as<typename V::value_type, uint8_t> ||
           std::same_as<typename V::value_type, int8_t>
inline float unroll4_sum_of_squares(const V& a) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float diff0 = a[i + 0];
    float diff1 = a[i + 1];
    float diff2 = a[i + 2];
    float diff3 = a[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    float diff0 = a[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

/**
 * Unrolled l2 distance between vector of float and vector of float
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float unroll4_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float diff0 = a[i + 0] - b[i + 0];
    float diff1 = a[i + 1] - b[i + 1];
    float diff2 = a[i + 2] - b[i + 2];
    float diff3 = a[i + 3] - b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    float diff0 = a[i + 0] - b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

/**
 * Unrolled l2 distance between vector of float and vector of uint8_t
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           (std::same_as<typename W::value_type, uint8_t> ||
            std::same_as<typename W::value_type, int8_t>)
inline float unroll4_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float diff0 = a[i + 0] - (float)b[i + 0];
    float diff1 = a[i + 1] - (float)b[i + 1];
    float diff2 = a[i + 2] - (float)b[i + 2];
    float diff3 = a[i + 3] - (float)b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    float diff0 = a[i + 0] - (float)b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

/**
 * Unrolled l2 distance between vector of uint8_t and vector of float
 */
template <feature_vector V, feature_vector W>
  requires(std::same_as<typename V::value_type, uint8_t> ||
           std::same_as<typename V::value_type, int8_t>) &&
          std::same_as<typename W::value_type, float>
inline float unroll4_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float diff0 = (float)a[i + 0] - b[i + 0];
    float diff1 = (float)a[i + 1] - b[i + 1];
    float diff2 = (float)a[i + 2] - b[i + 2];
    float diff3 = (float)a[i + 3] - b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    float diff0 = (float)a[i + 0] - b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

/**
 * Unrolled l2 distance between vector of uint8_t and vector of uint8_t
 */
template <feature_vector V, feature_vector W>
  requires(std::same_as<typename V::value_type, uint8_t> ||
           std::same_as<typename V::value_type, int8_t>) &&
          (std::same_as<typename W::value_type, uint8_t> ||
           std::same_as<typename W::value_type, int8_t>)
inline float unroll4_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float diff0 = (float)a[i + 0] - (float)b[i + 0];
    float diff1 = (float)a[i + 1] - (float)b[i + 1];
    float diff2 = (float)a[i + 2] - (float)b[i + 2];
    float diff3 = (float)a[i + 3] - (float)b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    float diff0 = (float)a[i + 0] - (float)b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

/****************************************************************
 *
 * Naive algorithms with start and stop
 *
 ****************************************************************/

/**
 * Compute l2 distance between two vectors of float
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float naive_sum_of_squares(
    const V& a, const W& b, size_t start, size_t stop) {
  float sum = 0.0;
  for (size_t i = start; i < stop; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

/**
 * Compute l2 distance between vector of float and vector of uint8_t
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           (std::same_as<typename W::value_type, uint8_t> ||
            std::same_as<typename W::value_type, int8_t>)
inline float naive_sum_of_squares(
    const V& a, const W& b, size_t start, size_t stop) {
  float sum = 0.0;
  for (size_t i = start; i < stop; ++i) {
    float diff = a[i] - (float)b[i];
    sum += diff * diff;
  }
  return sum;
}

/**
 * Compute l2 distance between vector of uint8_t and vector of float
 */
template <feature_vector V, feature_vector W>
  requires(std::same_as<typename V::value_type, uint8_t> ||
           std::same_as<typename V::value_type, int8_t>) &&
          std::same_as<typename W::value_type, float>
inline float naive_sum_of_squares(
    const V& a, const W& b, size_t start, size_t stop) {
  float sum = 0.0;
  for (size_t i = start; i < stop; ++i) {
    float diff = (float)a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

/**
 * Compute l2 distance between vector of uint8_t and vector of uint8_t
 */
template <feature_vector V, feature_vector W>
  requires(std::same_as<typename V::value_type, uint8_t> ||
           std::same_as<typename V::value_type, int8_t>) &&
          (std::same_as<typename W::value_type, uint8_t> ||
           std::same_as<typename W::value_type, int8_t>)
inline float naive_sum_of_squares(
    const V& a, const W& b, size_t start, size_t stop) {
  float sum = 0.0;
  for (size_t i = start; i < stop; ++i) {
    float diff = (float)a[i] - (float)b[i];
    sum += diff * diff;
  }
  return sum;
}

/****************************************************************
 *
 * 4x unrolled algorithms with start and stop. We have separate
 * functions despite the code duplication to make sure about
 * performance in the common case (no start / stop).
 *
 ****************************************************************/

/**
 * Unrolled l2 distance between vector of float and vector of float
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float unroll4_sum_of_squares(
    const V& a, const W& b, size_t begin, size_t end) {
  size_t loops = 4 * ((end - begin) / 4);
  size_t stop = begin + loops;
  float sum = 0.0;
  for (size_t i = begin; i < stop; i += 4) {
    float diff0 = a[i + 0] - b[i + 0];
    float diff1 = a[i + 1] - b[i + 1];
    float diff2 = a[i + 2] - b[i + 2];
    float diff3 = a[i + 3] - b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < end; ++i) {
    float diff0 = a[i + 0] - b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

/**
 * Unrolled l2 distance between vector of float and vector of uint8_t
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           (std::same_as<typename W::value_type, uint8_t> ||
            std::same_as<typename W::value_type, int8_t>)
inline float unroll4_sum_of_squares(
    const V& a, const W& b, size_t begin, size_t end) {
  size_t loops = 4 * ((end - begin) / 4);
  size_t stop = begin + loops;
  float sum = 0.0;
  for (size_t i = begin; i < stop; i += 4) {
    float diff0 = a[i + 0] - (float)b[i + 0];
    float diff1 = a[i + 1] - (float)b[i + 1];
    float diff2 = a[i + 2] - (float)b[i + 2];
    float diff3 = a[i + 3] - (float)b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < end; ++i) {
    float diff0 = a[i + 0] - (float)b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

/**
 * Unrolled l2 distance between vector of uint8_t and vector of float
 */
template <feature_vector V, feature_vector W>
  requires(std::same_as<typename V::value_type, uint8_t> ||
           std::same_as<typename V::value_type, int8_t>) &&
          std::same_as<typename W::value_type, float>
inline float unroll4_sum_of_squares(
    const V& a, const W& b, size_t begin, size_t end) {
  size_t loops = 4 * ((end - begin) / 4);
  size_t stop = begin + loops;
  float sum = 0.0;
  for (size_t i = begin; i < stop; i += 4) {
    float diff0 = (float)a[i + 0] - b[i + 0];
    float diff1 = (float)a[i + 1] - b[i + 1];
    float diff2 = (float)a[i + 2] - b[i + 2];
    float diff3 = (float)a[i + 3] - b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < end; ++i) {
    float diff0 = (float)a[i + 0] - b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

/**
 * Unrolled l2 distance between vector of uint8_t and vector of uint8_t
 */
template <feature_vector V, feature_vector W>
  requires(std::same_as<typename V::value_type, uint8_t> ||
           std::same_as<typename V::value_type, int8_t>) &&
          (std::same_as<typename W::value_type, uint8_t> ||
           std::same_as<typename W::value_type, int8_t>)
inline float unroll4_sum_of_squares(
    const V& a, const W& b, size_t begin, size_t end) {
  size_t loops = 4 * ((end - begin) / 4);
  size_t stop = begin + loops;
  float sum = 0.0;
  for (size_t i = begin; i < stop; i += 4) {
    float diff0 = (float)a[i + 0] - (float)b[i + 0];
    float diff1 = (float)a[i + 1] - (float)b[i + 1];
    float diff2 = (float)a[i + 2] - (float)b[i + 2];
    float diff3 = (float)a[i + 3] - (float)b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < end; ++i) {
    float diff0 = (float)a[i + 0] - (float)b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}

#endif  // TDB_SCORING_L2_DISTANCE_H
