

#include "detail/graph/coo.h"

#include <catch2/catch_all.hpp>
#include <set>
#include <vector>
#include "detail/linalg/vector.h"

TEST_CASE("coo: test test", "[coo]") {
  REQUIRE(true);
}

TEST_CASE("coo: test initializer list", "[coo]") {
  auto x = coo_matrix<int>{{0, 0, 1}, {1, 1, 2}, {2, 2, 3}};

  REQUIRE(x.nnz() == 3);

  CHECK(x(0) == std::tuple{0, 0, 1});
  CHECK(x(1) == std::tuple{1, 1, 2});
  CHECK(x(2) == std::tuple{2, 2, 3});

  auto y = coo_matrix<double, int>{{0, 0, 1}, {1, 1, 2}, {2, 2, 3}};

  CHECK(y(0) == std::tuple{0, 0, 1});
  CHECK(y(1) == std::tuple{1, 1, 2});
  CHECK(y(2) == std::tuple{2, 2, 3});
}

TEST_CASE("coo: test vector constructor", "[coo]") {
  auto x = coo_matrix<int, int>{
      Vector<int>{0, 1, 2}, Vector<int>{0, 1, 2}, Vector<int>{1, 2, 3}};
  auto y = coo_matrix<int>{
      Vector<size_t>{0, 1, 2}, Vector<size_t>{0, 1, 2}, Vector<int>{1, 2, 3}};
  auto z = coo_matrix<float>{
      Vector<size_t>{0, 1, 2}, Vector<size_t>{0, 1, 2}, Vector<float>{1, 2, 3}};

  REQUIRE(x.nnz() == 3);
  REQUIRE(y.nnz() == 3);
  REQUIRE(z.nnz() == 3);

  CHECK(x(0) == std::tuple{0, 0, 1});
  CHECK(x(1) == std::tuple{1, 1, 2});
  CHECK(x(2) == std::tuple{2, 2, 3});

  CHECK(y(0) == std::tuple{0, 0, 1});
  CHECK(y(1) == std::tuple{1, 1, 2});
  CHECK(y(2) == std::tuple{2, 2, 3});

  CHECK(z(0) == std::tuple{0, 0, 1});
  CHECK(z(1) == std::tuple{1, 1, 2});
  CHECK(z(2) == std::tuple{2, 2, 3});
}

TEST_CASE("coo: test vector constructor too", "[coo]") {
  auto a = Vector<int>{0, 1, 2};
  auto b = Vector<int>{0, 1, 2};
  auto c = Vector<int>{1, 2, 3};
  auto x = coo_matrix<int, int>{std::move(a), std::move(b), std::move(c)};
  CHECK(a.size() == 0);
  CHECK(b.size() == 0);
  CHECK(c.size() == 0);
  CHECK(a.data() == nullptr);
  CHECK(b.data() == nullptr);
  CHECK(c.data() == nullptr);
  CHECK(x.nnz() == 3);
}

TEST_CASE("coo: test move constructor", "[coo]") {
  auto x = coo_matrix<int, int>{
      Vector<int>{0, 1, 2}, Vector<int>{0, 1, 2}, Vector<int>{1, 2, 3}};
  auto y = coo_matrix{std::move(x)};

  CHECK(x.nnz() == 3);
  CHECK(y.nnz() == 3);

  // CHECK_THROWS(x(0));
  // CHECK_THROWS(x(1));
  // CHECK_THROWS(x(2));

  CHECK(y(0) == std::tuple{0, 0, 1});
  CHECK(y(1) == std::tuple{1, 1, 2});
  CHECK(y(2) == std::tuple{2, 2, 3});
}

TEST_CASE("coo: test tdb", "[coo]") {
  tiledb::Context ctx;
  std::string array_name =
      "/Users/lums/TileDB/coo2csr/external/csx/xout-placenta";
  auto x = tdb_coo_matrix<float, int64_t>(ctx, array_name);
}
