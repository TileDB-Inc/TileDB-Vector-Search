/**
 * @file   gen_queries.cc
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
 * @section DESCRIPTION
 *
 */

#include <catch2/catch_all.hpp>
#include <thread>
#include "utils/timer.h"

#include "flat_query.h"
#include "ivf_query.h"
#include "linalg.h"

bool global_debug = false;
std::string global_region = "us-east-1";

TEST_CASE("gen db", "[queries]") {
  size_t dimension = 128;
  tiledb::Context ctx;

  size_t db = GENERATE(1, 10, 100, 1000, 10000, 100000, 1000000, 10000000);

  std::cout << "db: " << db << std::endl;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-127, 127);

  auto db_mat = ColMajorMatrix<float>(dimension, db);
  for (auto& x : raveled(db_mat)) {
    x = dist(gen);
  }
  std::string db_name = "db_" + std::to_string(db) + ".tdb";
  write_matrix(ctx, db_mat, db_name);
}

TEST_CASE("gen q", "[queries]") {
  size_t dimension = 128;
  tiledb::Context ctx;

  size_t q = GENERATE(1, 10, 100, 1000, 10000);

  std::cout << "q: " << q << std::endl;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-127, 127);

  auto q_mat = ColMajorMatrix<float>(dimension, q);
  for (auto& x : raveled(q_mat)) {
    x = dist(gen);
  }
  std::string q_name = "q_" + std::to_string(q) + ".tdb";
  write_matrix(ctx, q_mat, q_name);
}