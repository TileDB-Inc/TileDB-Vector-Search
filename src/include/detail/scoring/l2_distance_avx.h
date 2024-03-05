/**
 * @file   l2_distance_avx.h
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
 * C++ functions for computing L2 distance between two feature vectors, using
 * AVX intrinsics
 *
 * @todo The loops can be unrolled to improve performance
 * @todo Implement "view" versions
 */

#ifndef TDB_SCORING_L2_DISTANCE_AVX_H
#define TDB_SCORING_L2_DISTANCE_AVX_H

#ifndef __AVX2__
// #warning "AVX2 not supported!! Using naive algorithms!!"
#else

#include <immintrin.h>
#include "concepts.h"

/**
 * float - float
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float avx2_sum_of_squares(const V& a, const W& b) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const float* a_ptr = a.data();
  const float* b_ptr = b.data();

  __m256 vec_sum = _mm256_setzero_ps();

  for (size_t i = start; i < stop; i += 8) {
    // Load 8 floats
    __m256 vec_a = _mm256_loadu_ps(a_ptr + i + 0);
    __m256 vec_b = _mm256_loadu_ps(b_ptr + i + 0);

    // Compute difference
    __m256 diff = _mm256_sub_ps(vec_a, vec_b);

    // Two alternatives for squaring and accumulating -- 2nd seems faster
#if 0
    // Compute square
    __m256 squared_diff = _mm256_mul_ps(diff, diff);

    // Accumulate
    vec_sum = _mm256_add_ps(vec_sum, squared_diff);
#else
    // Square and accumulate
    vec_sum = _mm256_fmadd_ps(diff, diff, vec_sum);
#endif
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
    float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

/**
 * float - uint8_t
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, uint8_t>
inline float avx2_sum_of_squares(const V& a, const W& b) {
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

    // Subtract floats
    __m256i diff = _mm256_sub_ps(a_floats, b_floats);

    // Square and add with fmadd
    vec_sum = _mm256_fmadd_ps(diff, diff, vec_sum);
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
    float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

/**
 * uint8_t - float
 */
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, float>
inline float avx2_sum_of_squares(const V& a, const W& b) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const uint8_t* a_ptr = a.data();
  const float* b_ptr = b.data();

  __m256 vec_sum = _mm256_setzero_ps();

  for (size_t i = start; i < stop; i += 8) {
    // Load 8 bytes
    __m128i vec_a = _mm_loadu_si64((__m64*)(a_ptr + i));

    // Load 8 floats
    __m256 b_floats = _mm256_loadu_ps(b_ptr + i + 0);

    // Zero extend 8bit to 32bit ints
    __m256i a_ints = _mm256_cvtepu8_epi32(vec_a);

    // Convert signed integers to floats
    __m256 a_floats = _mm256_cvtepi32_ps(a_ints);

    // Subtract floats
    __m256i diff = _mm256_sub_ps(a_floats, b_floats);

    // Square and add with fmadd
    vec_sum = _mm256_fmadd_ps(diff, diff, vec_sum);
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
    float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

/**
 * uint8_t - uint8_t
 */
template <class V, class W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, uint8_t>
inline float avx2_sum_of_squares(const V& a, const W& b) {
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

    // Two alternatives for computing difference - 2nd seems faster
#if 0
    // Convert signed integers to floats
    __m256 a_floats = _mm256_cvtepi32_ps(a_ints);
    __m256 b_floats = _mm256_cvtepi32_ps(b_ints);

    // Subtract floats
    __m256i diff = _mm256_sub_ps(a_floats, b_floats);
#else
    // Subtract signed integers
    __m256 i_diff = _mm256_sub_epi32(a_ints, b_ints);

    // Convert integers to floats
    __m256 diff = _mm256_cvtepi32_ps(i_diff);
#endif

    // Two alternatives for squaring and accumulating  -- 2nd seems faster
#if 0
    // Square and add in two steps
    __m256 diff_2 = _mm256_mul_ps(diff, diff);
    vec_sum = _mm256_add_ps(vec_sum, diff_2);

#else
    // Square and add with fmadd
    vec_sum = _mm256_fmadd_ps(diff, diff, vec_sum);
#endif
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
    float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

/**
 * float
 */
template <feature_vector V>
  requires std::same_as<typename V::value_type, float>
inline float avx2_sum_of_squares(const V& a) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const float* a_ptr = a.data();

  __m256 vec_sum = _mm256_setzero_ps();

  for (size_t i = start; i < stop; i += 8) {
    // Load 8 floats
    __m256 vec_a = _mm256_loadu_ps(a_ptr + i + 0);

    // Square and accumulate
    vec_sum = _mm256_fmadd_ps(vec_a, vec_a, vec_sum);
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
    sum += a[i] * a[i];
  }

  return sum;
}

/**
 * uint8_t
 */
template <class V>
  requires std::same_as<typename V::value_type, uint8_t>
inline float avx2_sum_of_squares(const V& a) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const uint8_t* a_ptr = a.data();

  __m256 vec_sum = _mm256_setzero_ps();

  for (size_t i = start; i < stop; i += 8) {
    // Load 8 bytes == 64 bits -- zeros out top 8 bytes
    __m128i vec_a = _mm_loadu_si64((__m64*)(a_ptr + i));

    // Zero extend 8bit to 32bit ints
    __m256i a_ints = _mm256_cvtepu8_epi32(vec_a);

    // Convert integers to floats
    __m256 a_floats = _mm256_cvtepi32_ps(a_ints);

    // Square and add with fmadd
    vec_sum = _mm256_fmadd_ps(a_floats, a_floats, vec_sum);
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
    sum += a[i] * a[i];
  }

  return sum;
}

#endif

#endif  // TDB_SCORING_L2_DISTANCE_AVX_H
