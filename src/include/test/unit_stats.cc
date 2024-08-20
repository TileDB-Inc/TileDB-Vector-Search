/**
 * @file   unit_stats.cc
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

#include "stats.h"

TEST_CASE("test build_config", "[stats]") {
  auto config = build_config();
  CHECK(!config.empty());
}

TEST_CASE("test logging_string and dump_logs", "[stats]") {
  { scoped_timer _("scoped_timer_1"); }
  log_timer _1{"log_timer_1"};
  _1.stop();
  log_timer _2{"log_timer_2"};
  _2.stop();

  std::string log = logging_string();
  std::cout << log << std::endl;
  REQUIRE(log.find("  scoped_timer_1: ") != std::string::npos);
  REQUIRE(log.find("log_timer_1: count: 1, ") != std::string::npos);
  REQUIRE(log.find("log_timer_2: count: 1, ") != std::string::npos);

  std::string path =
      (std::filesystem::temp_directory_path() / "dump_logs_file.txt").string();
  dump_logs(path, "IVF_PQ", 1, 2, 3, 4, 5.5f);
}
