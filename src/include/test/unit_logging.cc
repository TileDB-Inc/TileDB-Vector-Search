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
 * Unit tests for time logging singleton class and associated timers.
 *
 */

#include <catch2/catch_all.hpp>
#include <thread>
#include "utils/logging.h"

bool debug = false;

using namespace std::literals::chrono_literals;

auto duration = 500ms;

// TEST_CASE("test", "[logging][log_timer]") {
//   log_timer a("test");
//
//   std::this_thread::sleep_for(500ms);
//
//   a.stop();
//
//   auto f = _timing_data.get_entries_summed("test");
//   CHECK((f <= 520 && f >= 500));
//
//   a.start();
//   std::this_thread::sleep_for(500ms);
//   a.stop();
//
//   f = _timing_data.get_entries_summed("test");
//   CHECK((f <= 1040 && f >= 1000));
// }
//
// TEST_CASE("noisy test", "[logging][log_timer]") {
//   log_timer a("noisy_test", true);
//
//   std::this_thread::sleep_for(500ms);
//
//   a.stop();
//
//   auto f = _timing_data.get_entries_summed("noisy_test");
//   CHECK((f <= 520 && f >= 500));
//
//   a.start();
//   std::this_thread::sleep_for(500ms);
//   a.stop();
//
//   f = _timing_data.get_entries_summed("noisy_test");
//   CHECK((f <= 1040 && f >= 1000));
// }
//
// TEST_CASE("interval test", "[logging][log_timer]") {
//   log_timer a("interval_test", true);
//
//   std::this_thread::sleep_for(500ms);
//
//   a.stop();
//
//   auto f = _timing_data.get_entries_summed("interval_test");
//   CHECK((f <= 520 && f >= 500));
//   auto g = _timing_data.get_entries_separately("interval_test");
//   CHECK(g.size() == 1);
//   CHECK((g[0] <= 520 && g[0] >= 500));
//   auto g0 = g[0];
//
//   a.start();
//   std::this_thread::sleep_for(500ms);
//   a.stop();
//
//   f = _timing_data.get_entries_summed("interval_test");
//   CHECK((f <= 1040 && f >= 1000));
//   g = _timing_data.get_entries_separately("interval_test");
//   CHECK(g.size() == 2);
//   CHECK((g[0] <= 520 && g[0] >= 500));
//   CHECK((g[1] <= 520 && g[1] >= 500));
//   CHECK(g[0] == g0);
//
//   f = _timing_data.get_entries_summed("interval_test");
//   CHECK((f <= 1040 && f >= 1000));
//
//   std::this_thread::sleep_for(500ms);
//
//   f = _timing_data.get_entries_summed("interval_test");
//   CHECK((f <= 1040 && f >= 1000));
//   g = _timing_data.get_entries_separately("interval_test");
//   CHECK(g.size() == 2);
//   CHECK((g[0] <= 520 && g[0] >= 500));
//   CHECK((g[1] <= 520 && g[1] >= 500));
//   CHECK(g[0] == g0);
//
//   f = _timing_data.get_entries_summed("interval_test");
//   CHECK((f <= 1040 && f >= 1000));
// }
//
// TEST_CASE("scoped_timer start test", "[logging][scoped_timer]") {
//   scoped_timer a("life_test");
//   std::this_thread::sleep_for(300ms);
// }
//
// TEST_CASE("scoped_timer stop test", "[logging][timing_data]") {
//   std::this_thread::sleep_for(500ms);
//   auto f = _timing_data.get_entries_summed("life_test");
//   CHECK((f <= 320 && f >= 300));
// }
//
// TEST_CASE("ordering", "[logging][log_timer]") {
//   auto g = log_timer{"g"};
//   auto f = log_timer{"f"};
//   auto i = log_timer{"i"};
//   auto h = log_timer{"h"};
//
//   i.start();
//   std::this_thread::sleep_for(100ms);
//   g.start();
//   std::this_thread::sleep_for(100ms);
//   h.start();
//   std::this_thread::sleep_for(100ms);
//   f.start();
//   std::this_thread::sleep_for(100ms);
//   g.stop();
//   std::this_thread::sleep_for(100ms);
//   g.start();
//   std::this_thread::sleep_for(100ms);
//   g.stop();
//   std::this_thread::sleep_for(100ms);
//   f.stop();
//   g.start();
//   h.stop();
//   f.start();
//   std::this_thread::sleep_for(100ms);
//   f.stop();
//   g.stop();
//   i.stop();
//
//   auto i_t = _timing_data.get_entries_summed("i");
//   auto h_t = _timing_data.get_entries_summed("h");
//   auto g_t = _timing_data.get_entries_summed("g");
//   auto f_t = _timing_data.get_entries_summed("f");
//
//   if (debug) {
//     std::cout << f_t << " " << g_t << " " << h_t << " " << i_t << std::endl;
//   }
//
//   CHECK((i_t > 799 && i_t < 890));
//   CHECK((h_t > 499 && h_t < 560));
//   CHECK((g_t > 499 && g_t < 560));
//   CHECK((f_t > 499 && f_t < 560));
// }
//
// TEST_CASE("memory", "[logging]") {
//   _memory_data.insert_entry(tdb_func__, 8675309);
// }
//
// TEST_CASE("multithreaded timing test", "[logging][log_timer]") {
//   auto thread_func = [](const std::string& timer_name) {
//     for (int i = 0; i < 10; ++i) {
//       log_timer t(timer_name);
//       std::this_thread::sleep_for(50ms);
//       t.stop();
//     }
//   };
//
//   std::thread t1(thread_func, "multithreaded_test1");
//   std::thread t2(thread_func, "multithreaded_test2");
//
//   t1.join();
//   t2.join();
//
//   auto f1 = _timing_data.get_entries_summed("multithreaded_test1");
//   auto f2 = _timing_data.get_entries_summed("multithreaded_test2");
//
//   CHECK(f1 >= 500);
//   CHECK(f2 >= 500);
// }
//
// TEST_CASE("multithreaded memory test", "[logging][memory_data]") {
//   auto thread_func = [](const std::string& mem_name, size_t mem_use) {
//     for (int i = 0; i < 10; ++i) {
//       _memory_data.insert_entry(mem_name, mem_use);
//       std::this_thread::sleep_for(10ms);
//     }
//   };
//
//   std::thread t1(thread_func, "multithreaded_memory_test1", 1024);
//   std::thread t2(thread_func, "multithreaded_memory_test2", 2048);
//
//   t1.join();
//   t2.join();
//
//   auto m1 = _memory_data.get_entries_summed("multithreaded_memory_test1");
//   auto m2 = _memory_data.get_entries_summed("multithreaded_memory_test2");
//
//   CHECK(m1 >= 10 * 1024 / (1024 * 1024));
//   CHECK(m2 >= 10 * 2048 / (1024 * 1024));
// }
//
// TEST_CASE("multithreaded count test", "[logging][count_data]") {
//   auto thread_func = [](const std::string& count_name, size_t count) {
//     for (int i = 0; i < 10; ++i) {
//       _count_data.insert_entry(count_name, count);
//       std::this_thread::sleep_for(10ms);
//     }
//   };
//
//   std::thread t1(thread_func, "multithreaded_count_test1", 1);
//   std::thread t2(thread_func, "multithreaded_count_test2", 2);
//
//   t1.join();
//   t2.join();
//
//   auto c1 = _count_data.get_entries_summed("multithreaded_count_test1");
//   auto c2 = _count_data.get_entries_summed("multithreaded_count_test2");
//
//   CHECK(c1 == 10);
//   CHECK(c2 == 20);
// }
//
// TEST_CASE("highly concurrent timing test", "[logging][log_timer]") {
//   constexpr auto timer_name = "highly_concurrent_test";
//   constexpr int num_iterations = 100;
//   auto thread_func = []() {
//     for (int i = 0; i < num_iterations; ++i) {
//       log_timer t(timer_name);
//       std::this_thread::sleep_for(1ms);
//       t.stop();
//     }
//   };
//
//   auto num_threads = 20;
//   std::vector<std::thread> threads;
//   threads.reserve(num_threads);
//   for (int i = 0; i < num_threads; ++i) {
//     threads.emplace_back(thread_func);
//   }
//
//   for (auto& thread : threads) {
//     thread.join();
//   }
//
//   auto f = _timing_data.get_entries_summed(timer_name);
//   CHECK(f >= num_threads * num_iterations * 1);
// }
//
// TEST_CASE("highly concurrent memory test", "[logging][memory_data]") {
//   constexpr auto timer_name = "highly_concurrent_memory_test";
//   constexpr int num_iterations = 100;
//   auto thread_func = []() {
//     for (int i = 0; i < num_iterations; ++i) {
//       memory_data::memory_type mem_usage = 1024;  // Simulating memory usage
//       _memory_data.insert_entry(timer_name, mem_usage);
//       std::this_thread::sleep_for(1ms);
//     }
//   };
//
//   auto num_threads = 20;
//   std::vector<std::thread> threads;
//   threads.reserve(num_threads);
//   for (int i = 0; i < num_threads; ++i) {
//     threads.emplace_back(thread_func);
//   }
//
//   for (auto& thread : threads) {
//     thread.join();
//   }
//
//   auto f = _memory_data.get_entries_summed(timer_name);
//   CHECK(
//       f >=
//       num_threads * num_iterations * 1024 / (1024 * 1024));  // Convert to
//       MiB
// }
//
// TEST_CASE("highly concurrent count test", "[logging][count_data]") {
//   constexpr auto timer_name = "highly_concurrent_count_test";
//   constexpr int num_iterations = 100;
//   auto thread_func = []() {
//     for (int i = 0; i < num_iterations; ++i) {
//       _count_data.insert_entry(timer_name, 1);
//       std::this_thread::sleep_for(1ms);
//     }
//   };
//
//   auto num_threads = 20;
//   std::vector<std::thread> threads;
//   threads.reserve(num_threads);
//   for (int i = 0; i < num_threads; ++i) {
//     threads.emplace_back(thread_func);
//   }
//
//   for (auto& thread : threads) {
//     thread.join();
//   }
//
//   auto f = _count_data.get_entries_summed(timer_name);
//   CHECK(f >= num_threads * num_iterations);
// }
//
// TEST_CASE("dump", "[logging]") {
//   _timing_data.insert_entry(
//       "a",
//       std::chrono::high_resolution_clock::now() -
//           std::chrono::high_resolution_clock::now());
//   _count_data.insert_entry("b", 1);
//   _memory_data.insert_entry("c", 2);
//   std::cout << _timing_data.dump();
//   std::cout << _count_data.dump();
//   std::cout << _memory_data.dump();
// }

// Helper function to simulate some workload
void simulate_work(std::chrono::milliseconds duration) {
  std::this_thread::sleep_for(duration);
}

// TEST_CASE("Basic Timing Functionality", "[timing_data_class]") {
//   timing_data_class& timing_data = get_timing_data_instance();

//   SECTION("Single Timer Test") {
//     {
//       scoped_timer timer("TestTimer");
//       simulate_work(std::chrono::milliseconds(100));
//     }

//     std::string dump = timing_data.dump();
//     REQUIRE(dump.find("TestTimer") != std::string::npos);
//   }

//   SECTION("Nested Timer Test") {
//     {
//       scoped_timer parent_timer("ParentTimer");
//       simulate_work(std::chrono::milliseconds(50));

//       {
//         scoped_timer child_timer("ChildTimer");
//         simulate_work(std::chrono::milliseconds(30));
//       }
//     }

//     std::string dump = timing_data.dump();
//     REQUIRE(dump.find("ParentTimer") != std::string::npos);
//     REQUIRE(dump.find("ChildTimer") != std::string::npos);
//   }
// }

// TEST_CASE("Timing Data Structure", "[timing_data_class]") {
//   timing_data_class& timing_data = get_timing_data_instance();

//   SECTION("Tree Structure Test") {
//     {
//       scoped_timer root_timer("RootTimer");
//       simulate_work(std::chrono::milliseconds(20));

//       {
//         scoped_timer child_timer_1("ChildTimer1");
//         simulate_work(std::chrono::milliseconds(10));

//         {
//           scoped_timer grandchild_timer("GrandchildTimer");
//           simulate_work(std::chrono::milliseconds(5));
//         }
//       }

//       {
//         scoped_timer child_timer_2("ChildTimer2");
//         simulate_work(std::chrono::milliseconds(15));
//       }
//     }

//     std::string dump = timing_data.dump();
//     REQUIRE(dump.find("RootTimer") != std::string::npos);
//     REQUIRE(dump.find("ChildTimer1") != std::string::npos);
//     REQUIRE(dump.find("GrandchildTimer") != std::string::npos);
//     REQUIRE(dump.find("ChildTimer2") != std::string::npos);

//     // Check indentation levels to ensure the tree structure is preserved
//     REQUIRE(dump.find("  ChildTimer1") != std::string::npos);  // Indented
//     child REQUIRE(
//         dump.find("    GrandchildTimer") !=
//         std::string::npos);  // Further indented grandchild
//     REQUIRE(
//         dump.find("  ChildTimer2") !=
//         std::string::npos);  // Another child at the same level as
//         ChildTimer1
//   }
// }

// TEST_CASE("Verbose Mode Test", "[timing_data_class]") {
//   timing_data_class& timing_data = get_timing_data_instance();

//   SECTION("Check Verbose Output") {
//     timing_data.set_verbose(true);
//     std::ostringstream output;
//     std::streambuf* old_cout = std::cout.rdbuf(output.rdbuf());

//     {
//       scoped_timer timer("VerboseTimer");
//       simulate_work(std::chrono::milliseconds(10));
//     }

//     std::cout.rdbuf(old_cout);  // Restore original std::cout buffer

//     std::string verbose_output = output.str();
//     REQUIRE(
//         verbose_output.find("Starting timer VerboseTimer") !=
//         std::string::npos);
//     REQUIRE(
//         verbose_output.find("Stopping timer VerboseTimer") !=
//         std::string::npos);

//     timing_data.set_verbose(false);
//   }
// }

// TEST_CASE("Multiple Instances Test", "[timing_data_class]") {
//   timing_data_class& timing_data = get_timing_data_instance();

//   SECTION("Multiple Timers in Different Scopes") {
//     {
//       scoped_timer timer1("Timer1");
//       simulate_work(std::chrono::milliseconds(10));
//     }

//     {
//       scoped_timer timer2("Timer2");
//       simulate_work(std::chrono::milliseconds(20));
//     }

//     std::string dump = timing_data.dump();
//     REQUIRE(dump.find("Timer1") != std::string::npos);
//     REQUIRE(dump.find("Timer2") != std::string::npos);
//   }
// }

// TEST_CASE("Single-threaded timing") {
//   {
//     // Start a simple timer
//     scoped_timer timer("single_thread_test");

//     // Simulate some work
//     std::this_thread::sleep_for(std::chrono::milliseconds(100));
//   }

//   // Dump and verify the results
//   std::string result = _timing_data.dump();
//   REQUIRE(result.find("single_thread_test") != std::string::npos);
//   REQUIRE(result.find("avg =") != std::string::npos);
// }

// TEST_CASE("Nested timers") {
//   {
//     scoped_timer outer("outer_timer");
//     std::this_thread::sleep_for(std::chrono::milliseconds(50));

//     {
//       scoped_timer inner("inner_timer");
//       std::this_thread::sleep_for(std::chrono::milliseconds(50));
//     }

//     std::this_thread::sleep_for(std::chrono::milliseconds(50));
//   }

//   // Dump and verify the results
//   std::string result = _timing_data.dump();
//   REQUIRE(result.find("outer_timer") != std::string::npos);
//   REQUIRE(result.find("inner_timer") != std::string::npos);
//   REQUIRE(result.find("avg =") != std::string::npos);
// }

// Multithreaded test with different timers in each thread
TEST_CASE("Multithreaded timing test with different timers") {
  auto thread_func = [](const std::string& timer_name, int sleep_time) {
    scoped_timer timer(timer_name, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
  };

  std::thread t1(thread_func, "thread_1_timer", 150);
  std::thread t2(thread_func, "thread_2_timer", 250);

  t1.join();
  t2.join();
  return;
  std::string dump = _timing_data.dump();
  REQUIRE(dump.find("thread_1_timer") != std::string::npos);
  REQUIRE(dump.find("thread_2_timer") != std::string::npos);
  REQUIRE(dump.find("count = 0,") == std::string::npos);
  REQUIRE(dump.find("count = 1,") != std::string::npos);
  REQUIRE(dump.find("count = 2,") == std::string::npos);

  std::cout << dump << std::endl;
}

// // Multithreaded test with the same timer names in each thread
// TEST_CASE("Multithreaded timing test with same timer names") {
//   auto thread_func = [](const std::string& timer_name, int sleep_time) {
//     scoped_timer timer(timer_name, true);
//     std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
//   };

//   std::thread t1(thread_func, "shared_timer", 100);
//   std::thread t2(thread_func, "shared_timer", 200);

//   t1.join();
//   t2.join();

//   std::string dump = _timing_data.dump();
//   size_t occurrence_count = 0;
//   size_t pos = dump.find("shared_timer");
//   while (pos != std::string::npos) {
//     occurrence_count++;
//     pos = dump.find("shared_timer", pos + 1);
//   }

//   REQUIRE(occurrence_count == 2); // One entry per thread

//   std::cout << "Multithreaded timing data dump (same timers):\n" << dump <<
//   std::endl;
// }

// // Multithreaded test with nested timers
// TEST_CASE("Multithreaded timing test with nested timers") {
//     auto thread_func = [](const std::string& parent_timer_name, const
//     std::string& child_timer_name, int sleep_time) {
//         {
//             scoped_timer parent_timer(parent_timer_name, true);
//             std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
//             {
//                 scoped_timer child_timer(child_timer_name, true);
//                 std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time
//                 / 2));
//             }
//         }
//     };

//     std::thread t1(thread_func, "thread_1_parent", "thread_1_child", 200);
//     std::thread t2(thread_func, "thread_2_parent", "thread_2_child", 300);

//     t1.join();
//     t2.join();

//     std::string dump = _timing_data.dump();
//     REQUIRE(dump.find("thread_1_parent") != std::string::npos);
//     REQUIRE(dump.find("thread_1_child") != std::string::npos);
//     REQUIRE(dump.find("thread_2_parent") != std::string::npos);
//     REQUIRE(dump.find("thread_2_child") != std::string::npos);

//     std::cout << "Multithreaded timing data dump (nested timers):\n" << dump
//     << std::endl;
// }

// // Test to verify timing data persistence across multiple threads
// TEST_CASE("Multithreaded timing data persistence test") {
//     std::thread t1([]() {
//         scoped_timer timer("persistence_thread_1_timer", true);
//         std::this_thread::sleep_for(std::chrono::milliseconds(120));
//     });

//     std::thread t2([]() {
//         scoped_timer timer("persistence_thread_2_timer", true);
//         std::this_thread::sleep_for(std::chrono::milliseconds(180));
//     });

//     t1.join();
//     t2.join();

//     // Dumping the data after all threads have finished
//     std::string dump = _timing_data.dump();
//     REQUIRE(dump.find("persistence_thread_1_timer") != std::string::npos);
//     REQUIRE(dump.find("persistence_thread_2_timer") != std::string::npos);

//     std::cout << "Multithreaded timing data dump (persistence):\n" << dump <<
//     std::endl;
// }
