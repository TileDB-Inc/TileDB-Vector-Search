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
 * Very simple code for measuring details of program performance.  This file
defines
 * as singleton class that is used to log timing data.
 *
 * There are two timer classes associated with timing singleton.
 *
 * `log_timer`: A log_timer internally maintains a start time and a stop time
(which are
 * std::chrono::time_points).  It has three methods affecting timing: a
constructor, start(), and stop().
 * The constructor takes a string identifier and sets the start time to the
current time, using the
 * chrono high_resolution_clock.
 *
 * A new start time can be set by calling `start()`
 *
 * A call to `stop()` set the stop time to the current time and will record the
elapsed time
 * (the difference between the stop time and  most recent start time) in the
timing singleton,
 * using the string identifier passed to the constructor as the key.
 *
 * Thus start and stop should be used in pairs
 *
 * Each time stop() is called, a duration is separately recorded.  Thus the same
timer
 * can have associated with it multiple timing intervals.
 *
 * `scoped_timer`: Like the `log_timer` a scoped_timer internally maintains a
start time and a
 * stop time (which are std::chrono::time_points).  Unlike the `log_timer`,
timing is
 * contolled by the constructor and destructor.
 *
 * The constructor takes a string identifier and sets the start time to the
current time
 * The destructor records the time since the last start time in the timing
singleton, using
 * the string identifier passed to the constructor as the key.
 *
 * @note `start()` and `stop()` are currently available methods, but should not
be used with a scoped_timer
 *
 * The timing data is all available in the global singleton `_timing_data`.
 * * `get_timer_names()` returns a vector of the names of all timers that logged
data
 * * `get_entries_separately(const std::string&) returns a vector of all
intervals
 * recorded for that timer.  The units of the returned quantity are specified by
 * a template parameter (default is milliseconds).
 * * `get_intervals_summed(const std::string&)`
 * returns the cumulative time for all the intervals recorded for that timer.
 * The units of the returned quantity are specified by a template parameter
 * (default is milliseconds).
 *
 * This file also contains the definition of classes for logging memory
 * usage.  It does not automatically log memory usage, but provides a
 * mechanism to allow the user to log memory usage at particular points in the
code.
 * The associated singleton provides methods for getting memory usages
individually
 * or cumulatively.
 *
 * @example
 * @code{.cpp}
// Print totals for all timers
auto timers = _timing_data.get_timer_names();
for (const auto& timer : timers) {
   std::cout << timer << ":  " <<
_timing_data.get_intervals_summed<std::chrono::milliseconds>(timer) << " ms\n";
}
 * @endcode
 */

#ifndef TDB_LOGGING_COUNT_H
#define TDB_LOGGING_COUNT_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

/**
 * Singleton class for recording miscellaneous counts.  Internally maintains a
 * multimap of names and count values.  The `insert_entry()` method
 * is used to record the count.  The `get_entries_*()` methods are used
 * to query the recorded counts.
 *
 * Data are stored as integral values.  The `get_entries_*()` methods return
 * integral values.
 */
class count_data {
 public:
  using count_type = size_t;

 private:
  std::multimap<std::string, count_type> count_usages_;
  mutable std::mutex mtx_;
  bool verbose_{false};

  /**
   * Constructor.  Private to enforce singleton pattern.
   */
  count_data() = default;
  ~count_data() = default;

 public:
  /**
   * Copy constructor.  Deleted to enforce singleton pattern.
   */
  count_data(const count_data&) = delete;
  count_data& operator=(const count_data&) = delete;

  /**
   * Get a reference to the singleton instance of the class.
   * @return Reference to the singleton instance of the class.
   */
  static count_data& get_instance() {
    static std::once_flag flag;
    // This will leak, but it's okay - it's the Trusty Leaky Singleton pattern.
    static count_data* instance;
    std::call_once(flag, []() { instance = new count_data(); });
    return *instance;
  }

  /**
   * Insert a count entry into the multimap.
   * @param name The name to be associated with the count.
   * @param use The count to be recorded (in bytes).
   */
  void insert_entry(const std::string& name, const count_type& use) {
    std::lock_guard<std::mutex> lock(mtx_);
    count_usages_.insert(std::make_pair(name, use));
  }

  /**
   * Get the count entries associated with a name.
   * @param string Name to be queried.
   * @return Vector of count values associated with the name.
   */
  auto get_entries_separately(const std::string& string) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<double> usages;

    auto range = count_usages_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      usages.push_back(i->second);
    }
    return usages;
  }

  /**
   * Get the sum of the count entries associated with a name.
   * @param string Name to be queried.
   * @return Vector of count values associated with the name.
   */
  auto get_entries_summed(const std::string& string) const {
    std::lock_guard<std::mutex> lock(mtx_);
    double sum = 0.0;
    auto range = count_usages_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      sum += i->second;
    }
    return sum;
  }

  /**
   * Get the names associated with the count entries.
   * @return Vector of names associated with the count entries.
   */
  std::vector<std::string> get_usage_names() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::set<std::string> multinames;

    std::vector<std::string> names;

    for (const auto& i : count_usages_) {
      multinames.insert(i.first);
    }
    for (const auto& i : multinames) {
      names.push_back(i);
    }
    return names;
  }

  void set_verbose(bool verbose) {
    verbose_ = verbose;
  }

  bool get_verbose() const {
    return verbose_;
  }

  std::string dump() const {
    std::ostringstream oss;
    auto usage_names = get_usage_names();
    for (const auto& name : usage_names) {
      auto usages = get_entries_separately(name);
      double sum = get_entries_summed(name);
      double average = sum / usages.size();

      oss << name << ":";
      oss << " count: " << usages.size() << ",";
      oss << " sum: " << sum << ",";
      oss << " avg: " << average;
      oss << "\n";
    }
    return oss.str();
  }
};

inline count_data& get_count_data_instance() {
  return count_data::get_instance();
}

static count_data& _count_data{get_count_data_instance()};

#if 0
class stats_data {
  public:
    enum operation_type {open_array, submit_query};

    struct stats_type {
      operation_type operation;
      std::string uri;
      std::string location;
    };

    using stats_map = std::multimap<stats_type, std::string>;

  private:

    stats_map stats_;
    bool verbose_{false};

    /**
     * Constructor.  Private to enforce singleton pattern.
     */
    stats_data() = default;
    ~stats_data() = default;

  public:

    /**
     * Copy constructor.  Deleted to enforce singleton pattern.
     */
    stats_data(const stats_data&) = delete;
    stats_data& operator=(const stats_data&) = delete;

    /**
     * Get a reference to the singleton instance of the class.
     * @return Reference to the singleton instance of the class.
     */
    static stats_data& get_instance() {
      static stats_data instance;
      return instance;
    }

    /**
     * Insert a memory consumption entry into the multimap.
     * @param name The name to be associated with the memory consumption.
     * @param use The memory consumption to be recorded (in bytes).
     */
    void insert_entry(const stats_type& key, const std::string& stats) {
      stats_.insert({key, stats});
    }

    /**
     * Get the memory consumption entries associated with a name.
     * @param string Name to be queried.
     * @return Vector of memory consumption values associated with the name.
     */
    auto get_entries_separately(const std::string& string) const {
      std::vector<double> usages;

      auto range = stats_.equal_range(string);
      for (auto i = range.first; i != range.second; ++i) {
        usages.push_back(i->second / (1024*1024));
      }
      return usages;
    }
};
#endif

#endif  // TDB_LOGGING_COUNT_H
