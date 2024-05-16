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

// #include "scoring.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/vector.h"
#include "test/utils/randomize.h"
#include "utils/timer.h"

#include "detail/scoring/inner_product.h"
#include "detail/scoring/l2_distance.h"

#ifdef __AVX2__
#include <immintrin.h>
#include "detail/scoring/inner_product_avx.h"
#include "detail/scoring/l2_distance_avx.h"
#endif

double the_sum = 0.0;

auto baseline_scoring(
    const ColMajorMatrix<float>& a,
    const ColMajorMatrix<float>& b,
    size_t trials = 32) {
  life_timer _{tdb_func__, false};

  auto size_a = num_vectors(a);
  auto size_b = num_vectors(b);
  auto size_d = dimensions(a);
  const float* a_data = a.data();
  const float* b_data = b.data();

  the_sum = 0.0;
  for (size_t iter = 0; iter < trials; ++iter) {
    float sum = 0.0;
    for (size_t i = 0; i < size_a; ++i) {
      for (size_t j = 0; j < size_b; ++j) {
        for (size_t k = 0; k < size_d; ++k) {
          float diff = a_data[i * size_d + k] - b_data[j * size_d + k];
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
  requires std::invocable<
      F,
      std::span<typename V::value_type>,
      std::span<typename W::value_type>>
inline double do_time_scoring(
    const V& a, const W& b, F&& f, size_t trials = 32) {
  life_timer _{tdb_func__, false};
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

auto baseline_inner_product(
    const ColMajorMatrix<float>& a,
    const ColMajorMatrix<float>& b,
    size_t trials = 32) {
  life_timer _{tdb_func__, false};

  auto size_a = num_vectors(a);
  auto size_b = num_vectors(b);
  auto size_d = dimensions(a);
  const float* a_data = a.data();
  const float* b_data = b.data();

  the_sum = 0.0;
  for (size_t iter = 0; iter < trials; ++iter) {
    float sum = 0.0;
    for (size_t i = 0; i < size_a; ++i) {
      for (size_t j = 0; j < size_b; ++j) {
        for (size_t k = 0; k < size_d; ++k) {
          sum += a_data[i * size_d + k] * b_data[j * size_d + k];
        }
      }
    }
    the_sum += sum;
  }
  _.stop();
  return _.elapsed();
}

template <class V, class W>
inline double do_time_scoring_avx2(const V& a, const W& b, size_t trials = 32) {
  life_timer _{tdb_func__, false};
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

int main() {
  auto a = ColMajorMatrix<float>(128, 10'000);
  auto b = ColMajorMatrix<float>(128, 100);
  auto c = ColMajorMatrix<uint8_t>(128, 100);
  auto d = ColMajorMatrix<uint8_t>(128, 10'000);
  randomize(std::span<float>(a.data(), dimensions(a) * num_vectors(a)));
  randomize(std::span<float>(b.data(), dimensions(b) * num_vectors(b)));
  randomize(std::span<uint8_t>(c.data(), dimensions(c) * num_vectors(c)));
  randomize(std::span<uint8_t>(d.data(), dimensions(d) * num_vectors(d)));

  std::cout << "***** Inner Product" << std::endl;

  auto ta = baseline_inner_product(a, b);
  std::cout << "baseline inner product: " << ta << "s" << std::endl;
  auto inner_ = the_sum;

  {
    auto ta = do_time_scoring(
        a, b, naive_inner_product<decltype(a[0]), decltype(b[0])>);
    std::cout << "naive float-float inner_product: " << ta
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto tb = do_time_scoring(
        a, c, naive_inner_product<decltype(a[0]), decltype(c[0])>);
    std::cout << "naive float-uint8_t inner_product: " << tb
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto tc = do_time_scoring(
        c, a, naive_inner_product<decltype(c[0]), decltype(a[0])>);
    std::cout << "naive uint8_t-float inner_product: " << tc
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto td = do_time_scoring(
        c, d, naive_inner_product<decltype(c[0]), decltype(d[0])>);
    std::cout << "naive uint8_t-uint8_t inner_product: " << td
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;
  }

  {
    auto ta = do_time_scoring(
        a, b, unroll4_inner_product<decltype(a[0]), decltype(b[0])>);
    std::cout << "unrolled float-float inner_product: " << ta
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto tb = do_time_scoring(
        a, c, unroll4_inner_product<decltype(a[0]), decltype(c[0])>);
    std::cout << "unroll4 float-uint8_t inner_product: " << tb
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto tc = do_time_scoring(
        c, a, unroll4_inner_product<decltype(c[0]), decltype(a[0])>);
    std::cout << "unroll4 uint8_t-float inner_product: " << tc
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto td = do_time_scoring(
        c, d, unroll4_inner_product<decltype(c[0]), decltype(d[0])>);
    std::cout << "unroll4 uint8_t-uint8_t inner_product: " << td
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;
  }

#ifdef __AVX2__
  {
    auto ta = do_time_scoring(
        a, b, avx2_inner_product<decltype(a[0]), decltype(b[0])>);
    std::cout << "avx2 float-float inner_product: " << ta
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto tb = do_time_scoring(
        a, c, avx2_inner_product<decltype(a[0]), decltype(c[0])>);
    std::cout << "avx2 float -uint8_t inner_product: " << tb
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto tc = do_time_scoring(
        c, a, avx2_inner_product<decltype(c[0]), decltype(a[0])>);
    std::cout << "avx2 uint8_t-float inner_product: " << tc
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;

    auto td = do_time_scoring(
        c, d, avx2_inner_product<decltype(c[0]), decltype(d[0])>);
    std::cout << "avx2 uint8_t-uint8_t inner_product: " << td
              << "s -- inner_ - the_sum = " << inner_ - the_sum << std::endl;
  }
#endif

  std::cout << "***** L2 distance" << std::endl;

  auto t = baseline_scoring(a, b);
  std::cout << "baseline: " << t << "s" << std::endl;
  auto the_sum_ = the_sum;

  {
    auto t0 = do_time_scoring(
        a, b, [](auto&& a, auto&& b) { return naive_sum_of_squares(a, b); });
    std::cout << "naive float-float: " << t0
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;

#ifdef __AVX2__
    auto t4 = do_time_scoring(
        a, b, avx2_sum_of_squares<decltype(a[0]), decltype(b[0])>);
    std::cout << "avx2 float-float: " << t4
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;
#endif

    auto t1 = do_time_scoring(
        a, c, [](auto&& a, auto&& b) { return naive_sum_of_squares(a, b); });
    std::cout << "naive float-uint8_t: " << t1
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;

#ifdef __AVX2__
    auto t5 = do_time_scoring(
        a, c, avx2_sum_of_squares<decltype(a[0]), decltype(c[0])>);
    std::cout << "avx2 float-uint8_t: " << t5
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;
#endif

    auto t2 = do_time_scoring(
        c, a, [](auto&& a, auto&& b) { return naive_sum_of_squares(a, b); });
    std::cout << "naive uint8_t-float: " << t2
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;

#ifdef __AVX2__
    auto t6 = do_time_scoring(
        c, a, [](auto&& a, auto&& b) { return avx2_sum_of_squares(a, b); });
    std::cout << "avx2 uint8_t-float: " << t6
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;
#endif

    auto t3 = do_time_scoring(
        c, d, [](auto&& a, auto&& b) { return naive_sum_of_squares(a, b); });
    std::cout << "naive uint8_t-uint8_t: " << t3
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;

#ifdef __AVX2__
    auto t7 = do_time_scoring(
        c, d, [](auto&& a, auto&& b) { return avx2_sum_of_squares(a, b); });
    std::cout << "avx2 uint8_t-uint8_t: " << t7
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;
#endif
  }
  {
    auto t00 = do_time_scoring(
        a, b, [](auto&& a, auto&& b) { return unroll4_sum_of_squares(a, b); });
    std::cout << "unroll4 float-float: " << t00
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;

    auto t01 = do_time_scoring(
        a, c, [](auto&& a, auto&& b) { return unroll4_sum_of_squares(a, b); });
    std::cout << "unroll4 float-uint8_t: " << t01
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;

    auto t02 = do_time_scoring(
        c, a, [](auto&& a, auto&& b) { return unroll4_sum_of_squares(a, b); });
    std::cout << "unroll4 uint8_t-float: " << t02
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;

    auto t03 = do_time_scoring(
        c, d, [](auto&& a, auto&& b) { return unroll4_sum_of_squares(a, b); });
    std::cout << "unroll4 uint8_t-uint8_t: " << t03
              << "s -- the_sum_ - the_sum = " << the_sum_ - the_sum
              << std::endl;
  }
}
