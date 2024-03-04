/**
* @file   time_scoring.cc
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

#include "scoring.h"
#include "utils/timer.h"
#include "test_utils.h"
#include "detail/linalg/vector.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

double the_sum = 0.0;

auto baseline_scoring(const ColMajorMatrix<float>& a, const ColMajorMatrix<float>& b, size_t trials = 32) {
  life_timer _{tdb_func__, true};

  auto size_a = num_vectors(a);
  auto size_b = num_vectors(b);
  auto size_d = dimension(a);
  const float* a_data = a.data();
  const float* b_data = b.data();

  the_sum = 0.0;
  for (size_t iter = 0; iter < trials; ++iter) {
    float sum = 0.0;
    for (size_t i = 0; i < size_a; ++i) {
      for (size_t j = 0; j < size_b; ++j) {
	for (size_t k = 0; k < size_d; ++k) {
	  float diff = a_data[i*size_d+k] - b_data[j*size_d+k];
	  sum += diff * diff;
	}
      }
    }
    the_sum += sum;
  }
  _.stop();
  return _.elapsed();
}

template <class V, class W, class F>
inline double do_time_scoring(
    const V& a,
    const W& b,
    F&& f,
    size_t trials = 32) {
  life_timer _{tdb_func__,true};
  size_t size_a = num_vectors(a);
    size_t size_b = num_vectors(b);

  the_sum = 0.0;
  for (unsigned iter = 0; iter < trials; ++iter) {
    float sum = 0.0;
    for (unsigned i = 0; i < size_a; ++i) {
      for (unsigned j = 0; j < size_b; ++j) {
	sum += std::forward<F>(f)(a[i], b[j]);
      }
    }
    the_sum += sum;
  }
  _.stop();
  return _.elapsed();
}

template <class V, class W>
requires std::same_as<typename V::value_type, float> &&
    std::same_as<typename W::value_type, float>
inline float sum_of_squares_naive(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

template <class V, class W>
requires std::same_as<typename V::value_type, float> &&
    std::same_as<typename W::value_type, float>
inline float sum_of_squares_unroll4(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; i+=4) {
    float diff0 = a[i+0] - b[i+0];
    float diff1 = a[i+1] - b[i+1];
    float diff2 = a[i+2] - b[i+2];
    float diff3 = a[i+3] - b[i+3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }
  return sum;
}



template <class V, class W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, uint8_t>
inline float sum_of_squares_naive(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = a[i] - (float) b[i];
    sum += diff * diff;
  }
  return sum;
}

template <class V, class W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, float>
inline float sum_of_squares_naive(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff =(float) a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

template <class V, class W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, uint8_t>
inline float sum_of_squares_naive(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = (float) a[i] - (float) b[i];
    sum += diff * diff;
  }
  return sum;
}


#ifdef __AVX2__

template <class V, class W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float sum_of_squares_avx2(const V& a, const W& b) {

  // @todo Align on 256 bit boundaries 
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const float* a_ptr = a.data();
  const float* b_ptr = b.data();

  __m256 sum_vec = _mm256_setzero_ps();

  for (int i = start; i < stop; i += 8) {

  // @todo Align on 256 bit boundaries 
      __m256 vec_a = _mm256_loadu_ps(a_ptr + i + 0);
      __m256 vec_b = _mm256_loadu_ps(b_ptr + i + 0);

      __m256 diff = _mm256_sub_ps(vec_a, vec_b);
       sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);

       // __m256 squared_diff = _mm256_mul_ps(diff, diff);
       // sum_vec = _mm256_add_ps(sum_vec, squared_diff);
  }

    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 combined = _mm_add_ps(lo, hi);
    combined = _mm_hadd_ps(combined, combined);
    combined = _mm_hadd_ps(combined, combined);


    float sum = _mm_cvtss_f32(combined);

  for (size_t i = stop; i < size_a; ++i) {
    float diff = a[i] -	b[i];
    sum	+= diff	* diff;
  }

  return sum;
}


template <class V, class W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, uint8_t>
inline float sum_of_squares_avx2(const V& a, const W& b) {

  // @todo Align on 256 bit boundaries 
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const uint8_t* a_ptr = a.data();
  const uint8_t* b_ptr = b.data();

    size_t remaining = size_a;
    __m256 vec_sum = _mm256_setzero_ps();

    // Process arrays in chunks of 32 bytes (256 bits)
    while (remaining >=16) {
        // Load arrays into AVX2 registers
        __m128i vec_a = _mm_loadu_si128((__m128i*)a_ptr);
        __m128i vec_b = _mm_loadu_si128((__m128i*)b_ptr);

        // Convert to 16-bit integers
        __m256i a16 = _mm256_cvtepu8_epi16(vec_a);
        __m256i b16 = _mm256_cvtepu8_epi16(vec_b);

        // Subtract 'b' from 'a'
        __m256i idiff = _mm256_sub_epi16(a16, b16);

        // Convert to floating-point
        __m256 diff = _mm256_cvtepf32_ps(idiff);

        // Square the differences
        __m256 diff_2 = _mm256_mul_ps(diff, diff);

        // Accumulate squared differences
        vec_sum = _mm256_add_ps(vec_sum, diff_2);

        // Move to next chunk
        a += 16;
        b += 16;
        remaining -= 16;
    }

    // Horizontal addition to reduce to a single 256-bit vector
    __m128 lo = _mm256_castps256_ps128(vec_sum);
    __m128 hi = _mm256_extractf128_ps(vec_sum, 1);
    __m128 combined = _mm_add_ps(lo, hi);
    combined = _mm_hadd_ps(combined, combined);
    combined = _mm_hadd_ps(combined, combined);

    // Extract and return the result as a float
    float sum = _mm_cvtss_f32(combined);

    return sum;
}



template <class V, class W>
inline double do_time_scoring_avx2(
    const V& a,
    const W& b,
    size_t trials = 32) {
  life_timer _{tdb_func__,true};
  size_t size_a = num_vectors(a);
  size_t size_b = num_vectors(b);

  the_sum = 0.0;
  for (unsigned iter = 0; iter < trials; ++iter) {
    float sum = 0.0;
    for (unsigned i = 0; i < size_a; ++i) {
      for (unsigned j = 0; j < size_b; ++j) {
	sum += sum_of_squares_avx2(a[i], b[j]);
      }
    }
    the_sum += sum;
  }
  _.stop();
  return _.elapsed();
}



#endif

int main() {
  auto a = ColMajorMatrix<float>(128, 10'000);
  auto b = ColMajorMatrix<float>(128, 100);
  auto c = ColMajorMatrix<uint8_t>(128, 100);
  auto d = ColMajorMatrix<uint8_t>(128, 10'000);
  randomize(std::span<float>(a.data(), dimension(a) * num_vectors(a)));
  randomize(std::span<float>(b.data(), dimension(b) * num_vectors(b)));
  randomize(std::span<uint8_t>(c.data(), dimension(c) * num_vectors(c)));
  randomize(std::span<uint8_t>(d.data(), dimension(d) * num_vectors(d)));

  auto t = baseline_scoring(a, b);
  std::cout << "baseline: " << t << "s" << std::endl;
  auto the_sum_ = the_sum;

  auto t0 = do_time_scoring(a, b, sum_of_squares_naive<decltype(a[0]), decltype(b[0])>);
  std::cout << "naive float-float: " << t0 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

  auto t1 = do_time_scoring(a, c, sum_of_squares_naive<decltype(a[0]), decltype(c[0])>);
  std::cout << "naive float-uint8_t: " << t1 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

  auto t2 = do_time_scoring(c, a, sum_of_squares_naive<decltype(c[0]), decltype(a[0])>);
  std::cout << "naive uint8_t-float: " << t2 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

  auto t3 = do_time_scoring(c, d, sum_of_squares_naive<decltype(c[0]), decltype(d[0])>);
  std::cout << "naive uint8_t-uint8_t: " << t3 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

#ifdef __AVX2__
  auto t4 = do_time_scoring(a, b, sum_of_squares_avx2<decltype(a[0]), decltype(b[0])>);
  std::cout << "avx2 float-float: " << t4 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;
#endif

#if 0
  auto t5 = do_time_scoring(a, c, sum_of_squares_avx2<decltype(a[0]), decltype(c[0])>);
  std::cout << "avx2 float-uint8_t: " << t5 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

  auto t6 = do_time_scoring(c, a, sum_of_squares_avx2<decltype(c[0]), decltype(a[0])>);
  std::cout << "avx2 uint8_t-float: " << t6 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

  auto t7 = do_time_scoring(c, d, sum_of_squares_avx2<decltype(c[0]), decltype(d[0])>);
  std::cout << "avx2 uint8_t-uint8_t: " << t7 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;
#endif

#ifdef __AVX2__
  auto t8 = do_time_scoring_avx2(a, b);
  std::cout << "avx2 float-float: " << t8 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;
#endif

  auto t9 = do_time_scoring(a, b, sum_of_squares_unroll4<decltype(a[0]), decltype(b[0])>);
  std::cout << "unroll4 float-float: " << t9 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

}
