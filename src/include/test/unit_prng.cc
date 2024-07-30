/**
 * @file tiledb/common/random/test/unit_prng.cc
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
 * Tests for the global seedable PRNG facility.
 */

#include <catch2/catch_all.hpp>
#include <thread>
#include "utils/prng.h"
#include "utils/seeder.h"

TEST_CASE("Operator", "[prng]") {
  PRNG& prng = PRNG::get();
  // Verify that a second call succeeds.
  CHECK_NOTHROW(PRNG::get());
  auto rand_num1 = prng();
  CHECK(rand_num1 != 0);

  auto rand_num2 = prng();
  CHECK(rand_num2 != 0);
  CHECK(rand_num1 != rand_num2);

  auto rand_num3 = prng();
  CHECK(rand_num3 != 0);
  CHECK(rand_num1 != rand_num3);
  CHECK(rand_num2 != rand_num3);
}

TEST_CASE("Seeder singleton, errors", "[prng]") {
  /*
   * Retrieve a PRNG object explicitly. This will cause the PRNG to use the
   * singleton seeder, after which subsequent calls should fail.
   */
  [[maybe_unused]] auto& x{PRNG::get()};
  Seeder& seeder_ = Seeder::get();

  SECTION("try to set new seed after it's been set") {
    CHECK_THROWS_WITH(
        seeder_.set_seed(1),
        Catch::Matchers::ContainsSubstring("Seed has already been set"));
  }

  SECTION("try to use seed after it's been used") {
    CHECK_THROWS_WITH(
        seeder_.seed(),
        Catch::Matchers::ContainsSubstring("Seed can only be used once"));
  }
}

TEST_CASE("PRNG with different seeds generates different sequences", "[prng]") {
  PRNG& prng1 = PRNG::get();
  PRNG& prng2 = PRNG::get();

  std::vector<uint64_t> seq1;
  std::vector<uint64_t> seq2;

  for (int i = 0; i < 10; ++i) {
    seq1.push_back(prng1());
    seq2.push_back(prng2());
  }

  CHECK(seq1 != seq2);
}

TEST_CASE("Concurrent access to PRNG", "[prng]") {
  PRNG& prng = PRNG::get();
  const int num_threads = 10;
  const int num_iterations = 1000;
  std::vector<std::thread> threads;
  std::vector<uint64_t> results(num_threads * num_iterations);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&prng, &results, i]() {
      for (int j = 0; j < num_iterations; ++j) {
        results[i * num_iterations + j] = prng();
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& result : results) {
    CHECK(result != 0);
  }
}

TEST_CASE("Seeder random number testing", "[prng]") {
  PRNG& prng = PRNG::get();

  auto rand_num1 = prng();
  CHECK(rand_num1 != 0);

  auto rand_num2 = prng();
  CHECK(rand_num2 != 0);
  CHECK(rand_num1 != rand_num2);

  auto rand_num3 = prng();
  CHECK(rand_num3 != 0);
  CHECK(rand_num1 != rand_num3);
  CHECK(rand_num2 != rand_num3);
}

// Test PRNG with invalid Seeder usage
TEST_CASE("Invalid Seeder usage", "[prng]") {
  Seeder& seeder_ = Seeder::get();

  SECTION("try to set new seed after it's been set") {
    CHECK_THROWS_WITH(
        seeder_.set_seed(67890),
        Catch::Matchers::ContainsSubstring("Seed has already been set"));
  }

  SECTION("try to use seed after it's been used") {
    CHECK_THROWS_WITH(
        seeder_.seed(),
        Catch::Matchers::ContainsSubstring("Seed can only be used once"));
  }
}
