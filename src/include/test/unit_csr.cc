#include "coo.h"
#include "csr.h"
#include "timer.h"
#include <catch2/catch_all.hpp>
#include <set>
#include <vector>

#if HAS_TBB
#include <tbb/global_control.h>
#endif

TEST_CASE("csr: test test", "[csr]") {
  REQUIRE(true);
}

TEST_CASE("csr: test initializer list", "[csr]") {
  auto w = coo_matrix<double, int> {
    {1,  3, 0},
    { 2, 0, 1},
    { 1, 2, 2},
    { 5, 2, 3},
    { 0, 5, 4},
    { 5, 4, 5},
    { 0, 1, 6},
    { 5, 3, 7}
  };
  auto x = coo_matrix<double, int> {
    {0,  5, 4},
    { 0, 1, 6},
    { 1, 2, 2},
    { 1, 3, 0},
    { 2, 0, 1},
    { 5, 2, 3},
    { 5, 4, 5},
    { 5, 3, 7},
  };
  auto u = coo_matrix<double, int> {
    {0,  5, 4},
    { 0, 1, 6},
    { 1, 3, 0},
    { 1, 2, 2},
    { 2, 0, 1},
    { 5, 2, 3},
    { 5, 4, 5},
    { 5, 3, 7},
  };
  auto relabeled_x = coo_matrix<double, int> {
    {0,  5, 4},
    { 0, 1, 6},
    { 1, 2, 2},
    { 1, 3, 0},
    { 2, 0, 1},
    { 3, 2, 3},
    { 3, 3, 7},
    { 3, 4, 5},
  };
  auto relabeled_u = coo_matrix<double, int> {
    {0,  5, 4},
    { 0, 1, 6},
    { 1, 3, 0},
    { 1, 2, 2},
    { 2, 0, 1},
    { 3, 2, 3},
    { 3, 4, 5},
    { 3, 3, 7},
  };


  SECTION("move constructor") {
    auto y = csr_matrix(std::move(w));
    auto z = coo_matrix(std::move(y));

    for (size_t i = 0; i < z.nnz(); ++i) {
      if (z(i) != x(i)) {
        auto&& [z0, z1, z2] = z(i);
        auto&& [u0, u1, u2] = x(i);
        std::cout << i << " z: " << z0 << " " << z1 << " " << z2 << std::endl;
        std::cout << i << " x: " << u0 << " " << u1 << " " << u2 << std::endl;
      }
      CHECK(z(i) == x(i));
    }
  }
  SECTION("copy constructor") {
    auto y = csr_matrix(w);
    auto z = coo_matrix(std::move(y));
    for (size_t i = 0; i < z.nnz(); ++i) {
      if (z(i) != u(i)) {
        auto&& [z0, z1, z2] = z(i);
        auto&& [u0, u1, u2] = u(i);
        std::cout << i << " z: " << z0 << " " << z1 << " " << z2 << std::endl;
        std::cout << i << " u: " << u0 << " " << u1 << " " << u2 << std::endl;
      }
      CHECK(z(i) == u(i));
    }
  }

  SECTION("move constructor, relabeling") {
    auto y = csr_matrix(std::move(w), true);
    auto z = coo_matrix(std::move(y));

    for (size_t i = 0; i < z.nnz(); ++i) {
      if (z(i) != relabeled_x(i)) {
        auto&& [z0, z1, z2] = z(i);
        auto&& [u0, u1, u2] = relabeled_x(i);
        std::cout << i << " z: " << z0 << " " << z1 << " " << z2 << std::endl;
        std::cout << i << " rx: " << u0 << " " << u1 << " " << u2 << std::endl;
      }
      CHECK(z(i) == relabeled_x(i));
    }
  }

  SECTION("copy constructor, relabeling") {
    auto y = csr_matrix(w, true);
    auto z = coo_matrix(std::move(y));
    for (size_t i = 0; i < z.nnz(); ++i) {
      if (z(i) != relabeled_u(i)) {
        auto&& [z0, z1, z2] = z(i);
        auto&& [u0, u1, u2] = relabeled_u(i);
        std::cout << i << " z: " << z0 << " " << z1 << " " << z2 << std::endl;
        std::cout << i << " ru: " << u0 << " " << u1 << " " << u2 << std::endl;
      }
      CHECK(z(i) == relabeled_u(i));
    }
  }
}


#if 0
TEST_CASE("csr: info", "[csr]") {
  tiledb::Context ctx;
  std::string array_name = "/Users/lums/TileDB/coo2csr/external/csx/xout-placenta";
  // std::string array_name = "/home/lums/coo2csr/external/csx/xout-placenta";

  auto x = tdb_coo_matrix<float, int64_t> (ctx, array_name);
//  auto z = csr_matrix<float, int64_t> (x);
  auto w = csr_matrix<float, int64_t> (x, true);
}
#endif


template <class V>
void shuffle(V& v) {
  std::random_device rd;
  std::mt19937       g(rd());
  std::shuffle(v.begin(), v.end(), g);
}


template <class V>
void init_one(V& v) {
  size_t N = size(v);
  // size_t M = 1'500;
  // size_t repeats = ( N + M - 1) / M;
  size_t repeats = 1'500;
  size_t M       = (N + repeats - 1) / repeats;
  for (size_t i = 0; i < repeats; ++i) {
    std::iota(begin(v) + i * M, begin(v) + std::min((i + 1) * M, N), 1.01);
  }
}

coo_matrix<float, int64_t> new_coo(size_t N) {
  auto u = Vector<int64_t>(N);
  auto v = Vector<int64_t>(N);
  auto w = Vector<float>(N);
  init_one(u);
  init_one(v);
  init_one(w);
  return coo_matrix<float, int64_t>(std::move(u), std::move(v), std::move(w));
}

auto N = 100'000'000U;

#if HAS_TBB
TEST_CASE("csr: test tdb copy tbb, shuffle", "[csr]") {
  auto x = new_coo(N);

  for (auto&& max_p : { 1, 2, 4, 8 }) {
    auto _ = tbb::global_control(tbb::global_control::max_allowed_parallelism, max_p);

    x.shuffle_rows();

    std::cout << std::endl;
    scoped_timer t("csr copy tbb, shuffle", true);
    auto         y = csr_matrix<float, int64_t>(x, true);
  }

  std::cout << "--------------------------" << std::endl;
}

TEST_CASE("csr: test tdb copy tbb, no shuffle", "[csr]") {
  auto x = new_coo(N);

  for (auto&& max_p : { 1, 2, 4, 8 }) {
    std::cout << std::endl;

    scoped_timer t("csr copy tbb, no shuffle", true);
    auto         y = csr_matrix<float, int64_t>(x, true);
  }
  std::cout << "--------------------------" << std::endl;
}

TEST_CASE("csr: test tdb move tbb, shuffle", "[csr]") {
  for (auto&& max_p : { 1, 2, 4, 8 }) {
    auto _ = tbb::global_control(tbb::global_control::max_allowed_parallelism, max_p);

    auto x = new_coo(N);
    x.shuffle_rows();

    std::cout << std::endl;
    scoped_timer t("csr move tbb, shuffle", true);
    auto         y = csr_matrix<float, int64_t>(std::move(x), true);
  }
  std::cout << "--------------------------" << std::endl;
}

TEST_CASE("csr: test tdb move tbb, no shuffle", "[csr]") {
  for (auto&& max_p : { 1, 2, 4, 8 }) {
    auto _ = tbb::global_control(tbb::global_control::max_allowed_parallelism, max_p);

    auto x = new_coo(N);

    std::cout << std::endl;
    scoped_timer t("csr move tbb, no shuffle", true);
    auto         y = csr_matrix<float, int64_t>(std::move(x), true);
  }
  std::cout << "--------------------------" << std::endl;
}


#else
TEST_CASE("csr: test tdb copy tbb, shuffle", "[csr]") {
  auto x = new_coo(N);

  x.shuffle_rows();

  std::cout << std::endl;
  scoped_timer t("csr copy tbb, shuffle", true);
  auto         y = csr_matrix<float, int64_t>(x, true);


  std::cout << "--------------------------" << std::endl;
}

TEST_CASE("csr: test tdb copy tbb, no shuffle", "[csr]") {
  auto x = new_coo(N);

  std::cout << std::endl;

  scoped_timer t("csr copy tbb, no shuffle", true);
  auto         y = csr_matrix<float, int64_t>(x, true);
  std::cout << "--------------------------" << std::endl;
}

TEST_CASE("csr: test tdb move tbb, shuffle", "[csr]") {


  auto x = new_coo(N);
  x.shuffle_rows();

  std::cout << std::endl;
  scoped_timer t("csr move tbb, shuffle", true);
  auto         y = csr_matrix<float, int64_t>(std::move(x), true);

  std::cout << "--------------------------" << std::endl;
}

TEST_CASE("csr: test tdb move tbb, no shuffle", "[csr]") {

  auto x = new_coo(N);

  std::cout << std::endl;
  scoped_timer t("csr move tbb, no shuffle", true);
  auto         y = csr_matrix<float, int64_t>(std::move(x), true);
  std::cout << "--------------------------" << std::endl;
}

#endif


#if 0
TEST_CASE("csr: test tdb open", "[csr]") {
  tiledb::Context ctx;
  // std::string array_name = "/Users/lums/TileDB/coo2csr/external/csx/xout-placenta";
  std::string array_name = "/home/lums/coo2csr/external/csx/xout-placenta";

  auto x = tdb_coo_matrix<float, int64_t> (ctx, array_name);

}
#endif
