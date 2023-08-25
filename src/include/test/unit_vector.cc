

#include <catch2/catch_all.hpp>
#include <set>
#include <vector>
#include "detail/linalg/vector.h"

TEST_CASE("vector: test test", "[vector]") {
  REQUIRE(true);
}

TEST_CASE("vector: test constructor", "[vector]") {
  Vector<int> v(10);
  REQUIRE(v.num_rows() == 10);
  REQUIRE(v.size() == 10);
  REQUIRE(v.data() != nullptr);
}

TEST_CASE("vector: test move constructor", "[vector]") {
  Vector<int> v(10);
  auto p = v.data();
  Vector<int> w(std::move(v));
  REQUIRE(w.num_rows() == 10);
  REQUIRE(w.size() == 10);
  REQUIRE(w.data() == p);
}

TEST_CASE("vector: test operator()", "[vector]") {
  Vector<int> v(10);
  for (int i = 0; i < 10; ++i) {
    v(i) = i;
  }
  for (int i = 0; i < 10; ++i) {
    REQUIRE(v(i) == i);
  }
}

TEST_CASE("vector: test initializer_list", "[vector]") {
  std::vector<int> u{8, 6, 7, 5, 3, 0, 9};
  Vector<int> v{8, 6, 7, 5, 3, 0, 9};

  REQUIRE(v.num_rows() == 7);
  REQUIRE(v.size() == 7);
  REQUIRE(std::equal(v.data(), v.data() + 7, u.begin()));
  REQUIRE(std::equal(begin(v), end(v), u.begin()));
}

TEST_CASE("vector: test move constructor too", "[vector]") {
  std::vector<int> u{8, 6, 7, 5, 3, 0, 9};
  Vector<int> v{8, 6, 7, 5, 3, 0, 9};
  Vector<int> w{std::move(v)};

  REQUIRE(v.data() == nullptr);
  REQUIRE(w.size() == 7);
  REQUIRE(std::equal(begin(w), end(w), u.begin()));
}
