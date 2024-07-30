/**
 * @file random.h
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
 * @section DESCRIPTION
 *
 * Contains a random number generator which can be seeded.
 *
 */

#ifndef TILEDB_PRNG_HPP
#define TILEDB_PRNG_HPP

#include <mutex>
#include <random>

#include "utils/seeder.h"

namespace {
  /**
  //  * Implementation of the random seed.
  //  *
  //  * This is a class template in order to use `if constexpr`.
  //  *
  //  * @tparam return_size_type The type of the seed to be returned
  //  */
  // template <class return_size_type>
  // return_size_type random_seed() {
  //   static constexpr size_t rng_size = sizeof(std::random_device::result_type);
  //   static constexpr size_t ret_size = sizeof(return_size_type);
  //   std::random_device rng{};
  //   /*
  //   * We will need 64 bits to adequately seed the PRNG (`ret_size`). We support
  //   * cases where the result size of the RNG is 64 or 32 bits (`rng_size`).
  //   */
  //   if constexpr (ret_size == rng_size) {
  //     return rng();
  //   } else if constexpr (ret_size == 2 * rng_size) {
  //     return (rng() << rng_size) + rng();
  //   } else {
  //     throw std::runtime_error("Unsupported combination of RNG sizes");
  //   }
  // }

  // /**
  //  * The PRNG used within the random constructor.
  //  */
  // std::mt19937_64 prng_random() {
  //   return std::mt19937_64{random_seed<uint64_t>()};
  // }

  /**
   * The PRNG used within the default constructor.
   */
  std::mt19937_64 prng_default() {
    /*
    * Retrieve optional seed, which may or may not have been set explicitly.
    */
    auto seed{Seeder::get().seed()};
    /*
    * Use the seed if it has been set. Otherwise use a random seed.
    */
    if (seed.has_value()) {
      std::cout << "Using seed :" << seed.value() << std::endl;
      return std::mt19937_64{seed.value()};
    } else {
      // NOTE: If we wanted to have a default random seed, we could use this instead:
      // return prng_random();
      std::cout << "Using seed : 1234" << std::endl;
      return std::mt19937_64{1234};
    }
  }
}

class PRNG {
 public:
  /* ********************************* */
  /*     CONSTRUCTORS & DESTRUCTORS    */
  /* ********************************* */

  /**
   * Default constructor.
   *
   * If `Seeder` has been seeded, the seed will be set on the engine. Otherwise,
   * the generator is constructed with a default seed.
   */
  PRNG() : prng_(prng_default()), mtx_{} {}

  /** Copy constructor is deleted. */
  PRNG(const PRNG&) = delete;

  /** Move constructor is deleted. */
  PRNG(PRNG&&) = delete;

  /** Copy assignment is deleted. */
  PRNG& operator=(const PRNG&) = delete;

  /** Move assignment is deleted. */
  PRNG& operator=(PRNG&&) = delete;

  /** Destructor. */
  ~PRNG() = default;

  /* ********************************* */
  /*                API                */
  /* ********************************* */

  /** Singleton accessor. */
  static PRNG& get() {
    static PRNG singleton;
    return singleton;
  }

  /** Get next in PRNG sequence. */
  uint64_t operator()() {
    std::lock_guard<std::mutex> lock(mtx_);
    return prng_();
  }

  const std::mt19937_64 &generator() const {
    return prng_;
  }
  std::mt19937_64 &generator() {
    return prng_;
  }

 private:
  /* ********************************* */
  /*         PRIVATE ATTRIBUTES        */
  /* ********************************* */

  /** 64-bit mersenne twister engine for random number generation. */
  std::mt19937_64 prng_;

  /** Mutex which protects against simultaneous access to operator() body. */
  std::mutex mtx_;
};

#endif  // TILEDB_PRNG_HPP