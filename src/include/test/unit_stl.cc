

#include <catch2/catch_all.hpp>
#include <set>

TEST_CASE("basic set", "[set]") {
  std::set<int> s_a{1, 2, 3, 4, 5};
  CHECK(s_a.size() == 5);
  CHECK(*(s_a.begin()) == 1);

  std::set<int> s_d{5, 4, 3, 2, 1};
  CHECK(s_d.size() == 5);
  CHECK(*(s_d.begin()) == 1);
}

TEST_CASE("scalar set with less comparator", "[set]") {
  using set_type = std::set<size_t, std::less<size_t>>;

  set_type s_a{1, 2, 3, 4, 5};
  CHECK(s_a.size() == 5);
  CHECK(*(s_a.begin()) == 1);

  set_type s_d{5, 4, 3, 2, 1};
  CHECK(s_d.size() == 5);
  CHECK(*(s_d.begin()) == 1);
}

TEST_CASE("scalar set with greater comparator", "[set]") {
  using set_type = std::set<size_t, std::greater<size_t>>;

  set_type s_a{1, 2, 3, 4, 5};
  CHECK(s_a.size() == 5);
  CHECK(*(s_a.begin()) == 5);

  set_type s_d{5, 4, 3, 2, 1};
  CHECK(s_d.size() == 5);
  CHECK(*(s_d.begin()) == 5);
}
