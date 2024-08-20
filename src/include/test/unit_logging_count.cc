/**
 * @file unit_logging_count.cc
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
 * Unit tests for count logging singleton class.
 *
 */

#include <catch2/catch_all.hpp>
#include <thread>
#include "utils/logging_count.h"
#include "utils/logging_memory.h"
#include "utils/logging_time.h"

using namespace std::literals::chrono_literals;

TEST_CASE("multithreaded count test", "[logging_count]") {
  auto thread_func = [](const std::string& count_name, size_t count) {
    for (int i = 0; i < 10; ++i) {
      _count_data.insert_entry(count_name, count);
      std::this_thread::sleep_for(10ms);
    }
  };

  std::thread t1(thread_func, "multithreaded_count_test1", 1);
  std::thread t2(thread_func, "multithreaded_count_test2", 2);

  t1.join();
  t2.join();

  auto c1 = _count_data.get_entries_summed("multithreaded_count_test1");
  auto c2 = _count_data.get_entries_summed("multithreaded_count_test2");

  CHECK(c1 == 10);
  CHECK(c2 == 20);
}

TEST_CASE("highly concurrent count test", "[logging_count]") {
  constexpr auto timer_name = "highly_concurrent_count_test";
  constexpr int num_iterations = 100;
  auto thread_func = []() {
    for (int i = 0; i < num_iterations; ++i) {
      _count_data.insert_entry(timer_name, 1);
      std::this_thread::sleep_for(1ms);
    }
  };

  auto num_threads = 20;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  auto f = _count_data.get_entries_summed(timer_name);
  CHECK(f >= num_threads * num_iterations);
}

TEST_CASE("dump", "[logging_count]") {
  _count_data.insert_entry("b", 1);
  _memory_data.insert_entry("c", 2);
  std::cout << _count_data.dump();
  std::cout << _memory_data.dump();
}
