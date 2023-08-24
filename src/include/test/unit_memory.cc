

#include <memory>
#include <catch2/catch_all.hpp>

TEST_CASE("memory: test test", "[memory]") {
  REQUIRE(true);
}


TEST_CASE("memory: move unique_ptr scalar", "[memory]") {
  auto p = std::make_unique<int>(42);
  auto q = std::move(p);
  REQUIRE(*q == 42);
  REQUIRE(p == nullptr);
}

TEST_CASE("memory: move unique_ptr array", "[memory]") {
  std::unique_ptr<double[]> storage_{new double[42]};
  CHECK(std::equal(storage_.get(), storage_.get() + 42, std::vector<double>(42, 0.0).begin()));
  auto p = std::move(storage_);
  CHECK(p != nullptr);
  CHECK(storage_ == nullptr);
  CHECK(std::equal(p.get(), p.get() + 42, std::vector<double>(42, 0.0).begin()));
}