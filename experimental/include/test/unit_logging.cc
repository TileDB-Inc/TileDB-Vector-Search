/**
* @file   unit_logging.cc
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

#include <thread>
#include <chrono>
#include <catch2/catch_all.hpp>
#include "utils/logging.h"

using namespace::fixed_width_logging;
using namespace std::literals::chrono_literals;


auto duration = 500ms;

TEST_CASE("logging: test test", "[logging]") {
 REQUIRE(true);
}

TEST_CASE("logging: test", "[logging]") {
 log_timer a("test");

 std::this_thread::sleep_for(500ms);

  a.stop();

  auto f = _timing_data.get_intervals_summed("test");
  CHECK((f <= 510 && f >= 490));

  a.start();
  std::this_thread::sleep_for(500ms);
  a.stop();

  f = _timing_data.get_intervals_summed("test");
  CHECK((f <= 1010 && f >= 990));
}



TEST_CASE("logging: noisy test", "[logging]") {

  log_timer a("noisy_test", true);

  std::this_thread::sleep_for(500ms);

  a.stop();

  auto f = _timing_data.get_intervals_summed("noisy_test");
  CHECK((f <= 510 && f >= 490));

  a.start();
  std::this_thread::sleep_for(500ms);
  a.stop();

  f = _timing_data.get_intervals_summed("noisy_test");
  CHECK((f <= 1010 && f >= 990));
}



TEST_CASE("logging: interval test", "[logging]") {

  log_timer a("interval_test", true);

  std::this_thread::sleep_for(500ms);

  a.stop();

  auto f = _timing_data.get_intervals_summed("interval_test");
  CHECK((f <= 510 && f >= 490));
  auto g = _timing_data.get_intervals_separately("interval_test");
  CHECK(g.size() == 1);
  CHECK((g[0] <= 510 && g[0] >= 490));
  auto g0 = g[0];

  a.start();
  std::this_thread::sleep_for(500ms);
  a.stop();

  f = _timing_data.get_intervals_summed("interval_test");
  CHECK((f <= 1010 && f >= 990));
  g = _timing_data.get_intervals_separately("interval_test");
  CHECK(g.size() == 2);
  CHECK((g[0] <= 510 && g[0] >= 490));
  CHECK((g[1] <= 510 && g[1] >= 490));
  CHECK(g[0] == g0);

  f = _timing_data.get_intervals_summed("interval_test");
  CHECK((f <= 1010 && f >= 990));

  std::this_thread::sleep_for(500ms);

  f = _timing_data.get_intervals_summed("interval_test");
  CHECK((f <= 1010 && f >= 990));
  g = _timing_data.get_intervals_separately("interval_test");
  CHECK(g.size() == 2);
  CHECK((g[0] <= 510 && g[0] >= 490));
  CHECK((g[1] <= 510 && g[1] >= 490));
  CHECK(g[0] == g0);

  f = _timing_data.get_intervals_summed("interval_test");
  CHECK((f <= 1010 && f >= 990));
}
