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

auto baseline_scoring(const ColMajorMatrix<float>& a, const ColMajorMatrix<float>& b, size_t trials = 128) {
  life_timer _{tdb_func__};

  auto size_a = num_vectors(a) * dimension(a);
  const auto a_data = a.data();
  const auto b_data = b.data();

  the_sum = 0.0;
  for (unsigned iter = 0; iter < trials; ++iter) {
    float sum = 0.0;
    for (unsigned i = 0; i < size_a; ++i) {
      auto diff = a_data[i] - b_data[i];
      sum += diff * diff;
    }
    the_sum += sum;
  }
  _.stop();
  return _.elapsed();
}

template <class V, class W, class F>
auto do_time_scoring(
    const V& a,
    const W& b,
    F&& f,
    size_t trials = 128) {
  life_timer _{tdb_func__};

  the_sum = 0.0;
  for (unsigned iter = 0; iter < trials; ++iter) {
    float sum = 0.0;
    for (unsigned i = 0; i < num_vectors(a); ++i) {
      sum += std::forward<F>(f)(a[i], b[i]);
    }
    the_sum += sum;
  }
  _.stop();
  return _.elapsed();
}

template <class V, class W>
requires std::same_as<typename V::value_type, float> &&
    std::same_as<typename W::value_type, float>
inline auto sum_of_squares_naive(const V& a, const W& b) {
  auto sum = 0.0;
  for (size_t i = 0; i < dimension(a); ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

template <class V, class W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, uint8_t>
inline auto sum_of_squares_naive(const V& a, const W& b) {
  auto sum = 0.0;
  for (size_t i = 0; i < dimension(a); ++i) {
    float diff = a[i] - (float) b[i];
    sum += diff * diff;
  }
  return sum;
}

template <class V, class W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, float>
inline auto sum_of_squares_naive(const V& a, const W& b) {
  auto sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff =(float) a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

template <class V, class W>
  requires std::same_as<typename V::value_type, uint8_t> &&
           std::same_as<typename W::value_type, uint8_t>
inline auto sum_of_squares_naive(const V& a, const W& b) {
  auto sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = (float) a[i] * (float) b[i];
    sum += diff * diff;
  }
  return sum;
}


#ifdef __AVX2__

// Function to compute Euclidean distance between two points
float computeDistance(const float* a, const float* b, int size) {
  __m256 sum = _mm256_setzero_ps(); // Initialize sum to zero

  // Unroll the loop by 8
  for (int i = 0; i < size; i += 8) {
    // Load points into AVX registers
    __m256 aVec = _mm256_loadu_ps(a + i);
    __m256 bVec = _mm256_loadu_ps(b + i);

    // Subtract 'b' from 'a'
    __m256 diff = _mm256_sub_ps(aVec, bVec);

    // Square the differences
    __m256 squaredDiff = _mm256_mul_ps(diff, diff);

    // Accumulate squared differences
    sum = _mm256_add_ps(sum, squaredDiff);
  }

  // Horizontal addition of elements
  sum = _mm256_hadd_ps(sum, sum);
  sum = _mm256_hadd_ps(sum, sum);

  // Extract result and return
  return _mm_cvtss_f32(_mm256_castps256_ps128(sum));
}
#endif

int main() {
  auto a = ColMajorMatrix<float>(128, 1'000'000);
  auto b = ColMajorMatrix<float>(128, 1'000'000);
  auto c = ColMajorMatrix<uint8_t>(128, 1'000'000);
  auto d = ColMajorMatrix<uint8_t>(128, 1'000'000);
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
}
