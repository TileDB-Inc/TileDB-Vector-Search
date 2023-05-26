/**
 * @file   logging.h
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
 * Very simple code for measuring details of program performance.
 */


#ifndef TDB_LOGGING_H
#define TDB_LOGGING_H

#include <chrono>
#include <iostream>
#include <map>
#include <string>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

class timing_data {
private:
  json timings;

public:
  void add_timing(const std::string& operation, double elapsedTime) {
    timings[operation] = elapsedTime;
  }

  auto get_timings() {
    return timings;
  }
};

timing_data& get_timing_data_instance() {
  static timing_data instance;
  return instance;
}

timing_data& _timing_data { get_timing_data_instance() };

void add_timing(const std::string& operation, double elapsedTime) {
  _timing_data.add_timing(operation, elapsedTime);
}

auto get_timings() {
  return _timing_data.get_timings();
}

#endif    //TDB_LOGGING_H