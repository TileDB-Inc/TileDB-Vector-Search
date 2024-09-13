/**
 * @file   unit_inner_product_distance.cc
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
 * Test the inner product distance functions
 */
#include <catch2/catch_all.hpp>
#include <iomanip>
#include "detail/linalg/vector.h"
#include "detail/scoring/inner_product.h"
#include "detail/scoring/inner_product_avx.h"

TEST_CASE("simple vectors", "[inner_product_distance]") {
  auto u = std::vector<uint8_t>{1, 2, 3, 4};
  auto v = std::vector<uint8_t>{5, 6, 7, 8};
  auto x = std::vector<float>{1, 2, 3, 4};
  auto y = std::vector<float>{5, 6, 7, 8};

  auto z = std::vector<float>{1, 0, 1, -1};

  auto a = naive_inner_product(x, y);
  CHECK(a == 70);

  auto b = unroll4_inner_product(x, y);
  CHECK(b == 70);

  auto c = naive_inner_product(x, v);
  CHECK(c == 70);

  auto d = unroll4_inner_product(u, y);
  CHECK(d == 70);

  auto e = naive_inner_product(u, v);
  CHECK(e == 70);

  auto f = unroll4_inner_product(u, v);
  CHECK(f == 70);

  auto g = naive_inner_product(u, z);
  CHECK(g == 0);

  auto h = unroll4_inner_product(u, z);
  CHECK(h == 0);

  auto i = naive_inner_product(x, z);
  CHECK(i == 0);

  auto j = unroll4_inner_product(x, z);
  CHECK(j == 0);

#ifdef __AVX2__
  auto k = avx2_inner_product(x, y);
  CHECK(k == 70);

  auto l = avx2_inner_product(x, z);
  CHECK(l == 0);

  auto m = avx2_inner_product(u, y);
  CHECK(m == 70);

  auto n = avx2_inner_product(u, z);
  CHECK(n == 0);
#endif
}

TEST_CASE("negative values", "[inner_product_distance]") {
  auto x = std::vector<float>{
      -126.f, -121.f, -101.f, 127.f,  45.f,   -73.f,  -119.f, -128.f, -51.f,
      -88.f,  -86.f,  -90.f,  -114.f, -75.f,  -11.f,  -108.f, -111.f, -125.f,
      -128.f, -128.f, -125.f, -52.f,  76.f,   -89.f,  -76.f,  -123.f, -120.f,
      -123.f, -119.f, -111.f, -82.f,  -57.f,  -126.f, -125.f, -112.f, -82.f,
      120.f,  78.f,   -15.f,  -123.f, -128.f, -128.f, -108.f, 26.f,   -43.f,
      62.f,   78.f,   -122.f, -78.f,  -125.f, -126.f, -107.f, -100.f, -94.f,
      106.f,  83.f,   4.f,    -96.f,  -119.f, -120.f, -123.f, -124.f, -110.f,
      90.f,   -98.f,  -122.f, -112.f, -87.f,  -112.f, -97.f,  5.f,    9.f,
      -101.f, -128.f, -106.f, 92.f,   37.f,   -128.f, -124.f, -87.f,  25.f,
      -60.f,  -85.f,  55.f,   -39.f,  -126.f, -121.f, -93.f,  78.f,   -19.f,
      -117.f, -106.f, -60.f,  -102.f, -114.f, -78.f,  -43.f,  -86.f,  -103.f,
      -107.f, -128.f, -128.f, -91.f,  -40.f,  24.f,   -79.f,  -99.f,  93.f,
      -83.f,  -116.f, -124.f, -77.f,  -106.f, 0.f,    5.f,    60.f,   -73.f,
      -106.f, -124.f, -124.f, -78.f,  -84.f,  -111.f, -118.f, -59.f,  -48.f,
      -18.f,  -105.f};
  auto y = std::vector<float>{
      -59.f,  -74.f,  -101.f, -128.f, -66.f,  -51.f,  -115.f, -112.f, -110.f,
      -122.f, -128.f, -125.f, -59.f,  43.f,   32.f,   -88.f,  -70.f,  -25.f,
      -99.f,  -126.f, -115.f, -80.f,  13.f,   -61.f,  -125.f, -76.f,  -108.f,
      -112.f, -86.f,  -52.f,  26.f,   -125.f, -24.f,  -105.f, -109.f, -120.f,
      127.f,  -47.f,  -90.f,  -28.f,  -112.f, -126.f, -126.f, -98.f,  107.f,
      81.f,   -3.f,   -76.f,  15.f,   -98.f,  -122.f, -124.f, -122.f, -54.f,
      96.f,   74.f,   -89.f,  -120.f, -108.f, -106.f, -125.f, -67.f,  6.f,
      -80.f,  -45.f,  -112.f, -114.f, -72.f,  -12.f,  -75.f,  -100.f, -47.f,
      -120.f, -115.f, -29.f,  81.f,   102.f,  -68.f,  -126.f, -128.f, 32.f,
      15.f,   -1.f,   -56.f,  -119.f, -124.f, -124.f, -72.f,  -69.f,  -34.f,
      -73.f,  -84.f,  -123.f, -128.f, -121.f, -89.f,  -23.f,  -2.f,   -78.f,
      -79.f,  -104.f, -75.f,  -118.f, -113.f, -108.f, -81.f,  51.f,   10.f,
      -27.f,  53.f,   -55.f,  -100.f, -86.f,  41.f,   61.f,   -107.f, -121.f,
      -102.f, -17.f,  -48.f,  -120.f, -55.f,  -79.f,  23.f,   -92.f,  -122.f,
      -84.f,  -100.f};
  CHECK(x.size() == y.size());
  debug_vector(x, "x");
  debug_vector(y, "y");
  auto n = x.size();
  float manual = 0.f;
  for (size_t i = 0; i < n; ++i) {
    manual += x[i] * y[i];
  }
  std::cout << "manual: " << manual << std::endl;

  auto naive_xy = naive_inner_product(x, y);
  CHECK(std::abs(naive_xy - manual) < 0.01);
  auto unroll4_xy = unroll4_inner_product(x, y);
  CHECK(std::abs(unroll4_xy - manual) < 0.01);

  std::cout << "naive: " << naive_xy << std::endl;
  std::cout << "unroll4: " << unroll4_xy << std::endl;

#ifdef __AVX2__
  auto avx2_xy = avx2_inner_product(x, y);
  CHECK(std::abs(avx2_xy - manual) < 0.01);
  std::cout << "avx2: " << avx2_xy << std::endl;
#endif
}

TEST_CASE("simple longer vectors", "[inner_product_distance]") {
  std::cout << std::fixed << std::setprecision(7);

  size_t n = 100;
  auto x = std::vector<float>(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = i * 1.1f;
  }
  auto y = std::vector<float>(n);
  for (size_t i = 0; i < n; ++i) {
    y[i] = i * 10.f;
  }
  debug_vector(x, "x");
  debug_vector(y, "y");

  float manual = 0.f;
  for (size_t i = 0; i < n; ++i) {
    manual += x[i] * y[i];
  }
  std::cout << "manual:  " << manual << std::endl;

  auto naive_xy = naive_inner_product(x, y);
  auto unroll4_xy = unroll4_inner_product(x, y);

  std::cout << "naive:   " << naive_xy << std::endl;
  CHECK(std::abs(naive_xy - manual) < 0.01);
  std::cout << "unroll4: " << unroll4_xy << std::endl;
  CHECK(std::abs(unroll4_xy - manual) < 0.01);

#ifdef __AVX2__
  auto avx2_xy = avx2_inner_product(x, y);
  std::cout << "avx2: " << avx2_xy << std::endl;
#endif
}

TEMPLATE_TEST_CASE(
    "complex longer vectors", "[inner_product_distance]", int8_t, uint8_t) {
  // size_t n = GENERATE(1, 3, 127, 1021, 1024);

  // @todo generate a range of sizes (will require range of expected answers)
  size_t n = GENERATE(127);

  auto u = std::vector<TestType>(n);
  auto v = std::vector<TestType>(n);
  auto x = std::vector<float>(n);
  auto y = std::vector<float>(n);

  auto z = std::vector<float>(n);

  std::iota(begin(u), end(u), 0);
  std::iota(begin(v), end(v), 13);
  std::iota(begin(x), end(x), -3.14159);
  std::iota(begin(y), end(y), 17.8675309);

  std::iota(begin(z), end(z), 13);

  auto a = naive_inner_product(x, y);
  CHECK(std::abs(a - 785444.4375) < 0.01);
  auto ax = naive_inner_product(y, x);
  CHECK(ax == a);
  CHECK(std::abs(ax - 785444.4375) < 0.01);

  auto b = unroll4_inner_product(x, y);
  CHECK(std::abs(b - 785444.4375) < 0.01);
  CHECK(a == b);
  auto bx = unroll4_inner_product(y, x);
  CHECK(bx == b);
  CHECK(std::abs(bx - 785444.4375) < 0.01);

  auto a2 = naive_inner_product(u, v);
  CHECK(a2 == 778764);
  auto b2 = unroll4_inner_product(u, v);
  CHECK(b2 == 778764);
  CHECK(a2 == b2);

  auto a2x = naive_inner_product(v, u);
  CHECK(a2x == 778764);
  CHECK(a2 == a2x);
  auto b2x = unroll4_inner_product(v, u);
  CHECK(b2x == 778764);
  CHECK(a2x == b2x);

  auto a3 = naive_inner_product(u, x);
  CHECK(std::abs(a3 - 649615.1875) < 0.25);
  auto b3 = unroll4_inner_product(u, x);
  CHECK(std::abs(a3 - 649615.1875) < 0.25);
  CHECK(std::abs(a3 - b3) < 0.25);

  auto a3x = naive_inner_product(x, u);
  CHECK(std::abs(a3x - 649615.1875) < 0.25);
  CHECK(a3 == a3x);
  auto b3x = unroll4_inner_product(x, u);
  CHECK(std::abs(a3x - 649615.1875) < 0.25);
  CHECK(std::abs(a3x - b3x) < 0.25);

#ifdef __AVX2__
  {
    auto a = avx2_inner_product(x, y);
    CHECK(std::abs(a - 785444.4375) < 0.07);

    auto ax = avx2_inner_product(y, x);
    CHECK(ax == a);
    CHECK(std::abs(ax - 785444.4375) < 0.07);

    auto a2 = avx2_inner_product(u, v);
    CHECK(a2 == 778764);

    auto a2x = avx2_inner_product(v, u);
    CHECK(a2x == 778764);
    CHECK(a2 == a2x);

    auto a3 = avx2_inner_product(u, x);
    CHECK(std::abs(a3 - 649615.1875) < 0.25);

    auto a3x = avx2_inner_product(x, u);
    CHECK(std::abs(a3x - 649615.1875) < 0.25);
    CHECK(a3 == a3x);
  }
#endif

// @todo: inner_product does not yet have a partitioned version implemented yet.
// Leaving code here for future reference
#if 0
 // auto c = naive_inner_product(x, y, 0, size(x));
 // CHECK(c == 16 * n);

 auto d = unroll4_sum_of_squares(x, y, 0, size(x));
 CHECK(d == 16 * n);

 auto e = naive_sum_of_squares(x, y, 0, size(x) / 2);
 auto f = naive_sum_of_squares(x, y, size(x) / 2, size(x));
 CHECK(e == 16 * (size(x) / 2));
 CHECK(f == 16 * (size(x) - size(x) / 2));
 CHECK(e + f == 16 * n);

 auto g = unroll4_sum_of_squares(x, y, 0, size(x) / 2);
 auto h = unroll4_sum_of_squares(x, y, size(x) / 2, size(x));
 CHECK(g == 16 * (size(x) / 2));
 CHECK(h == 16 * (size(x) - size(x) / 2));

 CHECK(e == g);
 CHECK(f == h);

 CHECK(g + h == 16 * n);
#endif
}
