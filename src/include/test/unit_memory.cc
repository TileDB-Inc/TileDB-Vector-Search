

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <iostream>
#include <memory>
#include <vector>

TEST_CASE("move unique_ptr scalar", "[memory]") {
  auto p = std::make_unique<int>(42);
  auto q = std::move(p);
  REQUIRE(*q == 42);
  REQUIRE(p == nullptr);
}

TEST_CASE("move unique_ptr array", "[memory]") {
  // This will default initialize the array
  std::unique_ptr<double[]> storage_{new double[42]};

  // default-initialization does not zero-initialize
  std::fill(storage_.get(), storage_.get() + 42, 0.0);
  CHECK(std::equal(
      storage_.get(),
      storage_.get() + 42,
      std::vector<double>(42, 0.0).begin()));
  auto p = std::move(storage_);
  CHECK(p != nullptr);
  CHECK(storage_ == nullptr);
  CHECK(
      std::equal(p.get(), p.get() + 42, std::vector<double>(42, 0.0).begin()));

  auto x = std::vector<double>(42, 0.0);
  auto mm = std::mismatch(p.get(), p.get() + 42, begin(x));
  CHECK(mm.first == p.get() + 42);
  CHECK(mm.second == end(x));

  if (mm.first != p.get() + 42) {
    std::cout << "mismatch of p at " << mm.first - p.get() << std::endl;
  }
  if (mm.second != end(x)) {
    std::cout << "mismatch of x at " << mm.second - begin(x);
    std::cout << " x = " << *mm.second << std::endl;
  }
  auto l = std::find(p.get(), p.get() + 42, 0.0);
  CHECK(l == p.get());
  if (l != p.get()) {
    std::cout << "find of p at " << l - p.get() << std::endl;
  }
  auto m =
      std::find_if_not(p.get(), p.get() + 42, [](auto x) { return x == 0.0; });
  CHECK(m == p.get() + 42);
  if (m != p.get() + 42) {
    std::cout << "find_if_not of p at " << m - p.get();
    std::cout << " m = " << *m << std::endl;
  }
}
