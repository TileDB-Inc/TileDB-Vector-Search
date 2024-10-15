/**
 * @file tdb_catch_prng.h
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
 * This file declares a Catch2 hook to seed a global random number generator.
 * Based on:
 * - https://github.com/TileDB-Inc/TileDB/blob/dev/test/support/tdb_catch_prng.h
 */

#ifndef SET_SEED_FROM_CATCH_H
#define SET_SEED_FROM_CATCH_H

#include <catch2/catch_all.hpp>
#include <iostream>
#include "utils/prng.h"
#include "utils/seeder.h"

class SetSeedFromCatch : public Catch::EventListenerBase {
 public:
  /**
   * Make visible the base class constructor to default construct class
   * testPRNG using base class initialization.
   */
  using Catch::EventListenerBase::EventListenerBase;

  void testRunStarting(Catch::TestRunInfo const&) override {
    Seeder& seeder_ = Seeder::get();
    seeder_.set_seed(Catch::rngSeed());
  }
};

CATCH_REGISTER_LISTENER(SetSeedFromCatch)

#endif  //  SET_SEED_FROM_CATCH_H
