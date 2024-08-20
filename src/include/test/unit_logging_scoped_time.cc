/**
 * @file unit_logging_scoped_time.cc
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
#include "utils/logging_scoped_time.h"

bool debug = false;

using namespace std::literals::chrono_literals;

auto duration = 500ms;

TEST_CASE("scoped_timer start test", "[logging_scoped_timer]") {
  scoped_timer a("life_test");
  std::this_thread::sleep_for(300ms);
}

TEST_CASE("scoped_timer stop test", "[logging_scoped_timer]") {
  std::this_thread::sleep_for(500ms);
  auto f = _scoped_timing_data.get_entries_summed("life_test");
  CHECK((f <= 320 && f >= 300));
}

TEST_CASE("multithreaded timing test", "[logging_scoped_timer]") {
  auto thread_func = [](const std::string& timer_name) {
    for (int i = 0; i < 10; ++i) {
      scoped_timer t(timer_name);
      std::this_thread::sleep_for(50ms);
    }
  };

  std::thread t1(thread_func, "multithreaded_test1");
  std::thread t2(thread_func, "multithreaded_test2");

  t1.join();
  t2.join();

  auto f1 = _scoped_timing_data.get_entries_summed("multithreaded_test1");
  auto f2 = _scoped_timing_data.get_entries_summed("multithreaded_test2");

  CHECK(f1 >= 500);
  CHECK(f2 >= 500);
}

TEST_CASE("highly concurrent timing test", "[logging_scoped_timer]") {
  constexpr auto timer_name = "highly_concurrent_test";
  constexpr int num_iterations = 100;
  auto thread_func = []() {
    for (int i = 0; i < num_iterations; ++i) {
      scoped_timer t(timer_name);
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

  auto f = _scoped_timing_data.get_entries_summed(timer_name);
  CHECK(f >= num_threads * num_iterations * 1);
}

// Helper function to simulate some workload
void simulate_work(std::chrono::milliseconds duration) {
  std::this_thread::sleep_for(duration);
}

TEST_CASE("basic timing functionality", "[logging_scoped_timer]") {
  scoped_timing_data_class& timing_data = get_scoped_timing_data_instance();

  SECTION("single timer test") {
    {
      scoped_timer timer("TestTimer");
      simulate_work(std::chrono::milliseconds(100));
    }

    std::string dump = timing_data.dump();
    REQUIRE(dump.find("TestTimer") != std::string::npos);
  }

  SECTION("nested timer test") {
    {
      scoped_timer parent_timer("ParentTimer");
      simulate_work(std::chrono::milliseconds(50));

      {
        scoped_timer child_timer("ChildTimer");
        simulate_work(std::chrono::milliseconds(30));
      }
    }

    std::string dump = timing_data.dump();
    REQUIRE(dump.find("ParentTimer") != std::string::npos);
    REQUIRE(dump.find("ChildTimer") != std::string::npos);
  }
}

TEST_CASE("timing data structure", "[logging_scoped_timer]") {
  scoped_timing_data_class& timing_data = get_scoped_timing_data_instance();

  {
    scoped_timer root_timer("RootTimer");
    simulate_work(std::chrono::milliseconds(20));

    {
      scoped_timer child_timer_1("ChildTimer1");
      simulate_work(std::chrono::milliseconds(10));

      {
        scoped_timer grandchild_timer("GrandchildTimer");
        simulate_work(std::chrono::milliseconds(5));
      }
    }

    {
      scoped_timer child_timer_2("ChildTimer2");
      simulate_work(std::chrono::milliseconds(15));
    }
  }

  std::string dump = timing_data.dump();
  REQUIRE(dump.find("RootTimer") != std::string::npos);
  REQUIRE(dump.find("ChildTimer1") != std::string::npos);
  REQUIRE(dump.find("GrandchildTimer") != std::string::npos);
  REQUIRE(dump.find("ChildTimer2") != std::string::npos);

  // Check indentation levels to ensure the tree structure is preserved.
  // Indented (but not too indented).
  REQUIRE(dump.find("    ChildTimer1: count = 1,") != std::string::npos);
  REQUIRE(dump.find("     ChildTimer1: count = 1,") == std::string::npos);
  // Further indented grandchild (but not too indented).
  REQUIRE(dump.find("      GrandchildTimer: count = 1,") != std::string::npos);
  REQUIRE(dump.find("       GrandchildTimer: count = 1,") == std::string::npos);
  // Another child at the same level as the first child (but not too indented).
  REQUIRE(dump.find("    ChildTimer2: count = 1,") != std::string::npos);
  REQUIRE(dump.find("     ChildTimer2: count = 1,") == std::string::npos);
}

TEST_CASE("multiple Instances test", "[logging_scoped_timer]") {
  scoped_timing_data_class& timing_data = get_scoped_timing_data_instance();

  {
    scoped_timer timer1("Timer1");
    simulate_work(std::chrono::milliseconds(10));
  }

  {
    scoped_timer timer2("Timer2");
    simulate_work(std::chrono::milliseconds(20));
  }

  std::string dump = timing_data.dump();
  REQUIRE(dump.find("Timer1") != std::string::npos);
  REQUIRE(dump.find("Timer2") != std::string::npos);
}

TEST_CASE("single-threaded timing", "[logging_scoped_timer]") {
  {
    // Start a simple timer
    scoped_timer timer("single_thread_test");

    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Dump and verify the results
  std::string result = _scoped_timing_data.dump();
  REQUIRE(result.find("single_thread_test") != std::string::npos);
  REQUIRE(result.find("avg =") != std::string::npos);
}

TEST_CASE("nested timers", "[logging_scoped_timer]") {
  {
    scoped_timer outer("outer_timer");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    {
      scoped_timer inner("inner_timer");
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Dump and verify the results
  std::string result = _scoped_timing_data.dump();
  REQUIRE(result.find("outer_timer") != std::string::npos);
  REQUIRE(result.find("inner_timer") != std::string::npos);
  REQUIRE(result.find("avg =") != std::string::npos);
}

TEST_CASE("different timers in each thread", "[logging_scoped_timer]") {
  auto thread_func = [](const std::string& timer_name, int sleep_time) {
    scoped_timer timer(timer_name);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
  };

  std::thread t1(thread_func, "thread_1_timer", 50);
  std::thread t2(thread_func, "thread_2_timer", 75);
  std::thread t3(thread_func, "thread_3_timer", 100);

  t1.join();
  t2.join();
  t3.join();

  std::string dump = _scoped_timing_data.dump();
  REQUIRE(dump.find("thread_1_timer: count = 1") != std::string::npos);
  REQUIRE(dump.find("thread_2_timer: count = 1") != std::string::npos);
  REQUIRE(dump.find("thread_3_timer: count = 1") != std::string::npos);
  REQUIRE(dump.find("count = 0,") == std::string::npos);
  REQUIRE(dump.find("count = 1,") != std::string::npos);
  REQUIRE(dump.find("count = 2,") == std::string::npos);
  REQUIRE(dump.find("count = 3,") == std::string::npos);
}

TEST_CASE("same timer name in each thread", "[logging_scoped_timer]") {
  auto thread_func = [](const std::string& timer_name, int sleep_time) {
    scoped_timer timer(timer_name);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
  };

  std::thread t1(thread_func, "shared_timer", 100);
  std::thread t2(thread_func, "shared_timer", 200);

  t1.join();
  t2.join();

  std::string dump = _scoped_timing_data.dump();
  size_t occurrence_count = 0;
  size_t pos = dump.find("shared_timer: count = 1,");
  while (pos != std::string::npos) {
    occurrence_count++;
    pos = dump.find("shared_timer: count = 1,", pos + 1);
  }

  // One entry per thread, so two total.
  REQUIRE(occurrence_count == 2);
}

TEST_CASE("nested timers per thread", "[logging_scoped_timer]") {
  auto thread_func = [](const std::string& parent_timer_name,
                        const std::string& child_timer_name,
                        int sleep_time) {
    {
      scoped_timer parent_timer(parent_timer_name);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
      {
        scoped_timer child_timer(child_timer_name);
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time / 2));
      }
    }
  };

  std::thread t1(thread_func, "thread_1_parent", "thread_1_child", 200);
  std::thread t2(thread_func, "thread_2_parent", "thread_2_child", 300);

  t1.join();
  t2.join();

  std::string dump = _scoped_timing_data.dump();
  REQUIRE(
      dump.find("  thread_1_parent: count = 1, avg = ") != std::string::npos);
  REQUIRE(
      dump.find("   thread_1_parent: count = 1, avg = ") == std::string::npos);
  REQUIRE(
      dump.find("    thread_1_child: count = 1, avg = ") != std::string::npos);
  REQUIRE(
      dump.find("     thread_1_child: count = 1, avg = ") == std::string::npos);

  REQUIRE(
      dump.find("  thread_2_parent: count = 1, avg = ") != std::string::npos);
  REQUIRE(
      dump.find("   thread_2_parent: count = 1, avg = ") == std::string::npos);
  REQUIRE(
      dump.find("    thread_2_child: count = 1, avg = ") != std::string::npos);
  REQUIRE(
      dump.find("     thread_2_child: count = 1, avg = ") == std::string::npos);
}
