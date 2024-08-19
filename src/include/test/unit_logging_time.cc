/**
 * @file unit_logging_time.cc
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
 * Unit tests for time logging singleton class and associated timers.
 *
 */

#include <catch2/catch_all.hpp>
#include <thread>
#include "utils/logging_time.h"

bool debug = false;

using namespace std::literals::chrono_literals;

auto duration = 500ms;

TEST_CASE("test", "[logging_time]") {
  log_timer a("test");

  std::this_thread::sleep_for(500ms);

  a.stop();

  auto f = _timing_data.get_entries_summed("test");
  CHECK((f <= 520 && f >= 500));
  a.start();
  std::this_thread::sleep_for(500ms);
  a.stop();

  f = _timing_data.get_entries_summed("test");
  CHECK((f <= 1040 && f >= 1000));
}

TEST_CASE("noisy test", "[logging_time]") {
  log_timer a("noisy_test");

  std::this_thread::sleep_for(500ms);

  a.stop();

  auto f = _timing_data.get_entries_summed("noisy_test");
  CHECK((f <= 520 && f >= 500));

  a.start();
  std::this_thread::sleep_for(500ms);
  a.stop();

  f = _timing_data.get_entries_summed("noisy_test");
  CHECK((f <= 1040 && f >= 1000));
}

TEST_CASE("interval test", "[logging_time]") {
  log_timer a("interval_test");

  std::this_thread::sleep_for(500ms);

  a.stop();

  auto f = _timing_data.get_entries_summed("interval_test");
  CHECK((f <= 520 && f >= 500));
  auto g = _timing_data.get_entries_separately("interval_test");
  CHECK(g.size() == 1);
  CHECK((g[0] <= 520 && g[0] >= 500));
  auto g0 = g[0];

  a.start();
  std::this_thread::sleep_for(500ms);
  a.stop();

  f = _timing_data.get_entries_summed("interval_test");
  CHECK((f <= 1040 && f >= 1000));
  g = _timing_data.get_entries_separately("interval_test");
  CHECK(g.size() == 2);
  CHECK((g[0] <= 520 && g[0] >= 500));
  CHECK((g[1] <= 520 && g[1] >= 500));
  CHECK(g[0] == g0);

  f = _timing_data.get_entries_summed("interval_test");
  CHECK((f <= 1040 && f >= 1000));

  std::this_thread::sleep_for(500ms);

  f = _timing_data.get_entries_summed("interval_test");
  CHECK((f <= 1040 && f >= 1000));
  g = _timing_data.get_entries_separately("interval_test");
  CHECK(g.size() == 2);
  CHECK((g[0] <= 520 && g[0] >= 500));
  CHECK((g[1] <= 520 && g[1] >= 500));
  CHECK(g[0] == g0);

  f = _timing_data.get_entries_summed("interval_test");
  CHECK((f <= 1040 && f >= 1000));
}

TEST_CASE("ordering", "[logging_time]") {
  auto g = log_timer{"g"};
  auto f = log_timer{"f"};
  auto i = log_timer{"i"};
  auto h = log_timer{"h"};

  i.start();
  std::this_thread::sleep_for(100ms);
  g.start();
  std::this_thread::sleep_for(100ms);
  h.start();
  std::this_thread::sleep_for(100ms);
  f.start();
  std::this_thread::sleep_for(100ms);
  g.stop();
  std::this_thread::sleep_for(100ms);
  g.start();
  std::this_thread::sleep_for(100ms);
  g.stop();
  std::this_thread::sleep_for(100ms);
  f.stop();
  g.start();
  h.stop();
  f.start();
  std::this_thread::sleep_for(100ms);
  f.stop();
  g.stop();
  i.stop();

  auto i_t = _timing_data.get_entries_summed("i");
  auto h_t = _timing_data.get_entries_summed("h");
  auto g_t = _timing_data.get_entries_summed("g");
  auto f_t = _timing_data.get_entries_summed("f");

  if (debug) {
  }

  CHECK((i_t > 799 && i_t < 890));
  CHECK((h_t > 499 && h_t < 560));
  CHECK((g_t > 499 && g_t < 560));
  CHECK((f_t > 499 && f_t < 560));
}

TEST_CASE("multithreaded timing test", "[logging_time]") {
  auto thread_func = [](const std::string& timer_name) {
    for (int i = 0; i < 10; ++i) {
      log_timer t(timer_name);
      std::this_thread::sleep_for(50ms);
      t.stop();
    }
  };

  std::thread t1(thread_func, "multithreaded_test1");
  std::thread t2(thread_func, "multithreaded_test2");

  t1.join();
  t2.join();

  auto f1 = _timing_data.get_entries_summed("multithreaded_test1");
  auto f2 = _timing_data.get_entries_summed("multithreaded_test2");

  CHECK(f1 >= 500);
  CHECK(f2 >= 500);
}

TEST_CASE("highly concurrent timing test", "[logging_time]") {
  constexpr auto timer_name = "highly_concurrent_test";
  constexpr int num_iterations = 100;
  auto thread_func = []() {
    for (int i = 0; i < num_iterations; ++i) {
      log_timer t(timer_name);
      std::this_thread::sleep_for(1ms);
      t.stop();
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

  auto f = _timing_data.get_entries_summed(timer_name);
  CHECK(f >= num_threads * num_iterations * 1);
}
