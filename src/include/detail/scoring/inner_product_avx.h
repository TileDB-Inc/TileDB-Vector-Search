/**
 * @file   inner_product_avx.h
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
 * C++ functions for computing the inner product between two feature vectors,
 * using AVX2 intrinsics
 *
 */

#ifndef TDB_SCORING_INNER_PRODUCT_AVX_H
#define TDB_SCORING_INNER_PRODUCT_AVX_H

#ifndef __AVX2__
// #warning "AVX2 not supported!! Using naive algorithms!!"
#else

#include <immintrin.h>
#include "concepts.h"

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float avx2_inner_product(const V& a, const W& b) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const float* a_ptr = a.data();
  const float* b_ptr = b.data();

  __m256 vec_sum = _mm256_setzero_ps();

  for (size_t i = start; i < stop; i += 8) {
    // Load 8 floats
    __m256 vec_a = _mm256_loadu_ps(a_ptr + i);
    __m256 vec_b = _mm256_loadu_ps(b_ptr + i);

    // Multiply and accumulate
    vec_sum = _mm256_fmadd_ps(vec_a, vec_b, vec_sum);
  }

  // 8 to 4
  __m128 lo = _mm256_castps256_ps128(vec_sum);
  __m128 hi = _mm256_extractf128_ps(vec_sum, 1);
  __m128 combined = _mm_add_ps(lo, hi);

  // 4 to 2
  combined = _mm_hadd_ps(combined, combined);

  // 2 to 1
  combined = _mm_hadd_ps(combined, combined);

  float sum = _mm_cvtss_f32(combined);

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, uint8_t>
inline float avx2_inner_product(const V& a, const W& b) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const float* a_ptr = a.data();
  const uint8_t* b_ptr = b.data();

  __m256 vec_sum = _mm256_setzero_ps();

  for (size_t i = start; i < stop; i += 8) {
    // Load 8 floats
    __m256 a_floats = _mm256_loadu_ps(a_ptr + i + 0);

    // Load 8 bytes
    __m128i vec_b = _mm_loadu_si64((__m64*)(b_ptr + i));

    // Zero extend 8bit to 32bit ints
    __m256i b_ints = _mm256_cvtepu8_epi32(vec_b);

    // Convert signed integers to floats
    __m256 b_floats = _mm256_cvtepi32_ps(b_ints);

    // Multiply and accumulate
    vec_sum = _mm256_fmadd_ps(a_floats, b_floats, vec_sum);
  }

  // 8 to 4
  __m128 lo = _mm256_castps256_ps128(vec_sum);
  __m128 hi = _mm256_extractf128_ps(vec_sum, 1);
  __m128 combined = _mm_add_ps(lo, hi);

  // 4 to 2
  combined = _mm_hadd_ps(combined, combined);

  // 2 to 1
  combined = _mm_hadd_ps(combined, combined);

  float sum = _mm_cvtss_f32(combined);

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, float>
inline float avx2_inner_product(const V& a, const W& b) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const uint8_t* a_ptr = a.data();
  const float* b_ptr = b.data();

  __m256 vec_sum = _mm256_setzero_ps();

  for (size_t i = start; i < stop; i += 8) {
    // Load 8 bytes == 64 bits -- zeros out top 8 bytes
    __m128i vec_a = _mm_loadu_si64((__m64*)(a_ptr + i));

    // Load 8 floats
    __m256 b_floats = _mm256_loadu_ps(b_ptr + i + 0);

    // Zero extend 8bit to 32bit ints
    __m256i a_ints = _mm256_cvtepu8_epi32(vec_a);

    // Convert signed integers to floats
    __m256 a_floats = _mm256_cvtepi32_ps(a_ints);

    // Multiply and accumulate
    vec_sum = _mm256_fmadd_ps(a_floats, b_floats, vec_sum);
  }

  // 8 to 4
  __m128 lo = _mm256_castps256_ps128(vec_sum);
  __m128 hi = _mm256_extractf128_ps(vec_sum, 1);
  __m128 combined = _mm_add_ps(lo, hi);

  // 4 to 2
  combined = _mm_hadd_ps(combined, combined);

  // 2 to 1
  combined = _mm_hadd_ps(combined, combined);

  float sum = _mm_cvtss_f32(combined);

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}

template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, uint8_t>
inline float avx2_inner_product(const V& a, const W& b) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const uint8_t* a_ptr = a.data();
  const uint8_t* b_ptr = b.data();

  __m256 vec_sum = _mm256_setzero_ps();

  for (size_t i = start; i < stop; i += 8) {
    // Load 8 bytes == 64 bits -- zeros out top 8 bytes
    __m128i vec_a = _mm_loadu_si64((__m64*)(a_ptr + i));
    __m128i vec_b = _mm_loadu_si64((__m64*)(b_ptr + i));

    // Zero extend 8bit to 32bit ints
    __m256i a_ints = _mm256_cvtepu8_epi32(vec_a);
    __m256i b_ints = _mm256_cvtepu8_epi32(vec_b);

    // Convert signed integers to floats
    __m256 a_floats = _mm256_cvtepi32_ps(a_ints);
    __m256 b_floats = _mm256_cvtepi32_ps(b_ints);

    // Multiply and accumulate
    vec_sum = _mm256_fmadd_ps(a_floats, b_floats, vec_sum);
  }

  // 8 to 4
  __m128 lo = _mm256_castps256_ps128(vec_sum);
  __m128 hi = _mm256_extractf128_ps(vec_sum, 1);
  __m128 combined = _mm_add_ps(lo, hi);

  // 4 to 2
  combined = _mm_hadd_ps(combined, combined);

  // 2 to 1
  combined = _mm_hadd_ps(combined, combined);

  float sum = _mm_cvtss_f32(combined);

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}

#endif  // __AVX2__
#endif  // TDB_SCORING_INNER_PRODUCT_AVX_H
