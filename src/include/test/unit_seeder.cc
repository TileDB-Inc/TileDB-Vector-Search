/**
 * @file unit_seeder.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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
#include "utils/seeder.h"

TEST_CASE("Default seed", "[seeder]") {
  // Default seed is std::nullopt
  Seeder seeder;
  std::optional<uint64_t> seed;

  // Use default seed (state 0 -> 2)
  CHECK_NOTHROW(seed = seeder.seed());
  CHECK(!seed.has_value());

  // Try setting seed after it's been used (state 2)
  SECTION("try to set seed again") {
    CHECK_THROWS_WITH(
        seeder.set_seed(123),
        Catch::Matchers::ContainsSubstring("Seed has already been set"));
  }

  // Try using seed after it's been used (state 2)
  SECTION("try to use seed again") {
    CHECK_THROWS_WITH(
        seeder.seed(),
        Catch::Matchers::ContainsSubstring("Seed can only be used once"));
  }
}

TEST_CASE("Set seed", "[seeder]") {
  // Set seed (state 0 -> 1)
  Seeder seeder;
  CHECK_NOTHROW(seeder.set_seed(123));

  SECTION("try to set seed again") {
    CHECK_THROWS_WITH(
        seeder.set_seed(456),
        Catch::Matchers::ContainsSubstring("Seed has already been set"));
  }

  // Use seed, after it's been set but not used (state 1 -> 2)
  CHECK(seeder.seed() == 123);

  // Try setting seed after it's been set & used (state 2)
  SECTION("try to set seed after it's been set and used") {
    CHECK_THROWS_WITH(
        seeder.set_seed(456),
        Catch::Matchers::ContainsSubstring("Seed has already been set"));
  }

  // Try using seed after it's been set & used (state 2)
  SECTION("try to use seed after it's been set and used") {
    CHECK_THROWS_WITH(
        seeder.seed(),
        Catch::Matchers::ContainsSubstring("Seed can only be used once"));
  }
}

TEST_CASE("Concurrent access", "[seeder]") {
  Seeder seeder;
  std::optional<uint64_t> seed;
  std::thread t1([&]() { CHECK_NOTHROW(seeder.set_seed(999)); });

  std::thread t2([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK_THROWS_WITH(
        seeder.set_seed(888),
        Catch::Matchers::ContainsSubstring("Seed has already been set."));
  });

  t1.join();
  t2.join();

  CHECK(seeder.seed() == 999);

  // Ensure no race condition allowed seed to be set twice
  SECTION("verify seed set only once") {
    CHECK_THROWS_WITH(
        seeder.set_seed(777),
        Catch::Matchers::ContainsSubstring("Seed has already been set."));
  }
}

TEST_CASE("Singleton behavior", "[seeder]") {
  Seeder& seeder1 = Seeder::get();
  Seeder& seeder2 = Seeder::get();

  // Ensure seeder1 and seeder2 are the same instance
  CHECK(&seeder1 == &seeder2);

  // Check we cannot set the seed
  CHECK_THROWS_WITH(
      seeder1.set_seed(101),
      Catch::Matchers::ContainsSubstring("Seed has already been set."));

  // Ensure they have the same seed value
  auto seed = seeder1.seed();
  CHECK(seed.has_value());
  CHECK_THROWS_WITH(
      seeder2.seed(),
      Catch::Matchers::ContainsSubstring("Seed can only be used once"));

  // Verify singleton behavior: second instance cannot set a different seed
  CHECK_THROWS_WITH(
      seeder2.set_seed(202),
      Catch::Matchers::ContainsSubstring("Seed has already been set."));
}

TEST_CASE("State transition verification", "[seeder]") {
  Seeder seeder;

  // Default state (0)
  std::optional<uint64_t> seed;
  CHECK_NOTHROW(seed = seeder.seed());
  CHECK(!seed.has_value());

  // After default seed usage, transition to state (2)
  SECTION("verify state after default seed usage") {
    CHECK_THROWS_WITH(
        seeder.set_seed(123),
        Catch::Matchers::ContainsSubstring("Seed has already been set."));
  }

  // Reset the seeder to default state (for testing purposes only)
  Seeder seeder_reset;
  CHECK_NOTHROW(seeder_reset.set_seed(456));
  CHECK(seeder_reset.seed() == 456);

  // Verify state transition 1 -> 2
  CHECK_THROWS_WITH(
      seeder_reset.seed(),
      Catch::Matchers::ContainsSubstring("Seed can only be used once"));
}
