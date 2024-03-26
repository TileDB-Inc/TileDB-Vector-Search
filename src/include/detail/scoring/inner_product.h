/**
 * @file   inner_product.h
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
 * C++ functions for computing the inner product between two feature vectors
 *
 */

#ifndef TDB_SCORING_INNER_PRODUCT_H
#define TDB_SCORING_INNER_PRODUCT_H

#include "concepts.h"

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float naive_inner_product(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, uint8_t>
inline float naive_inner_product(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    sum += a[i] * ((float)b[i]);
  }
  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, float>
inline float naive_inner_product(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    sum += ((float)a[i]) * b[i];
  }
  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, uint8_t>
inline float naive_inner_product(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    sum += ((float)a[i]) * ((float)b[i]);
  }
  return sum;
}

/*******************************************************************************
 *
 * Unrolled inner products
 *
 ******************************************************************************/
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float unroll4_inner_product(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float prod0 = a[i + 0] * b[i + 0];
    float prod1 = a[i + 1] * b[i + 1];
    float prod2 = a[i + 2] * b[i + 2];
    float prod3 = a[i + 3] * b[i + 3];
    sum += prod0 + prod1 + prod2 + prod3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    sum += a[i + 0] * b[i + 0];
  }
  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, uint8_t>
inline float unroll4_inner_product(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float prod0 = a[i + 0] * ((float)b[i + 0]);
    float prod1 = a[i + 1] * ((float)b[i + 1]);
    float prod2 = a[i + 2] * ((float)b[i + 2]);
    float prod3 = a[i + 3] * ((float)b[i + 3]);
    sum += prod0 + prod1 + prod2 + prod3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    sum += a[i + 0] * ((float)b[i + 0]);
  }
  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, float>
inline float unroll4_inner_product(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float prod0 = ((float)a[i + 0]) * b[i + 0];
    float prod1 = ((float)a[i + 1]) * b[i + 1];
    float prod2 = ((float)a[i + 2]) * b[i + 2];
    float prod3 = ((float)a[i + 3]) * b[i + 3];
    sum += prod0 + prod1 + prod2 + prod3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    sum += ((float)a[i + 0]) * b[i + 0];
  }
  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, uint8_t>
inline float unroll4_inner_product(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float prod0 = ((float)a[i + 0]) * ((float)b[i + 0]);
    float prod1 = ((float)a[i + 1]) * ((float)b[i + 1]);
    float prod2 = ((float)a[i + 2]) * ((float)b[i + 2]);
    float prod3 = ((float)a[i + 3]) * ((float)b[i + 3]);
    sum += prod0 + prod1 + prod2 + prod3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    sum += ((float)a[i + 0]) * ((float)b[i + 0]);
  }
  return sum;
}

#endif  // TDB_SCORING_INNER_PRODUCT_H
