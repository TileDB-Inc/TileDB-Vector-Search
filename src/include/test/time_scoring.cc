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
    float diff = a[i] -	b[i];
    sum	+= diff	* diff;
  }

  return sum;
}


auto baseline_inner_product(const ColMajorMatrix<float>& a, const ColMajorMatrix<float>& b, size_t trials = 32) {
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
	  sum += a_data[i*size_d+k] * b_data[j*size_d+k];
	}
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
inline float inner_product_naive(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

template <class V, class W>
requires std::same_as<typename V::value_type, uint8_t> &&
std::same_as<typename W::value_type, uint8_t>
inline float inner_product_naive(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    sum +=((float) a[i]) * ((float)b[i]);
  }
  return sum;
}


template <class V, class W>
requires std::same_as<typename V::value_type, float> &&
std::same_as<typename W::value_type, float>
inline float inner_product_unroll4(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; i+=4) {
    float prod0 = a[i+0] * b[i+0];
    float prod1 = a[i+1] * b[i+1];
    float prod2 = a[i+2] * b[i+2];
    float prod3 = a[i+3] * b[i+3];
    sum += prod0 + prod1 + prod2 + prod3;
  }
  return sum;
}


template <class V, class W>
requires std::same_as<typename V::value_type, float> &&
std::same_as<typename W::value_type, float>
inline float inner_product_avx2(const V& a, const W& b) {

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

template <class V, class W>
requires std::same_as<typename V::value_type, uint8_t> &&
std::same_as<typename W::value_type, uint8_t>
inline float inner_product_avx2(const V& a, const W& b) {

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

  auto ta = baseline_inner_product(a, b);
  std::cout << "baseline inner product: " << ta << "s" << std::endl;
  auto inner_ = the_sum;

  auto tb = do_time_scoring(a, b, inner_product_naive<decltype(a[0]), decltype(b[0])>);
  std::cout << "naive float-float inner_product: " << tb << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

  auto tc = do_time_scoring(c, d, inner_product_naive<decltype(c[0]), decltype(d[0])>);
  std::cout << "naive uint8_t-uint8_t inner_product: " << tc << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

  auto td = do_time_scoring(a, b, inner_product_unroll4<decltype(a[0]), decltype(b[0])>);
  std::cout << "unrolled float-float inner_product: " << td << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

  auto te = do_time_scoring(a, b, inner_product_avx2<decltype(a[0]), decltype(b[0])>);
  std::cout << "avx2 float-float inner_product: " << te << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

  auto tf = do_time_scoring(c, d, inner_product_avx2<decltype(c[0]), decltype(d[0])>);
  std::cout << "avx2 uint8_t-uint8_t inner_product: " << tf << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;


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
  auto t8 = do_time_scoring(c, d, sum_of_squares_avx2<decltype(c[0]), decltype(d[0])>);
  std::cout << "avx2 uint8_t-uint8_t: " << t8 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

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


  auto t9 = do_time_scoring(a, b, sum_of_squares_unroll4<decltype(a[0]), decltype(b[0])>);
  std::cout << "unroll4 float-float: " << t9 << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum << std::endl;

}
