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

#ifndef TDB_LOGGING_H
#define TDB_LOGGING_H

#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

/**
 *  Macro holding the name of the function in which it is called.
 */
#ifndef tdb_func__
#ifdef __cpp_lib_source_location
#include <source_location>
#define tdb_func__                                  \
  std::string {                                     \
    std::source_location::current().function_name() \
  }
#else
#define tdb_func__ \
  std::string {    \
    (__func__)     \
  }
#endif
#endif

/**
 * Singleton class for storing timing data.
 *
 * As a singleton, the object from this class can be used from anywhere in the
 * code. Timing data is stored in a multimap, where the key string is the name
 * of the timer and the value is a duration.
 */
class timing_data_class {
 public:
  using clock_type = std::chrono::high_resolution_clock;
  using time_type = std::chrono::time_point<clock_type>;
  using duration_type =
      std::chrono::duration<clock_type::rep, clock_type::period>;

 private:
  std::multimap<std::string, duration_type> interval_times_;
  mutable std::mutex mtx_;
  bool verbose_{false};

  /**
   * Private constructor and destructor for singleton.
   */
  timing_data_class() = default;
  ~timing_data_class() = default;

 public:
  /**
   * Delete copy constructor and assignment operator.
   */
  timing_data_class(const timing_data_class&) = delete;
  timing_data_class& operator=(const timing_data_class&) = delete;

  /**
   * Return a reference to the singleton instance.
   * @return The singleton instance.
   */
  // static timing_data_class& get_instance() {
  //   static timing_data_class instance;
  //   return instance;
  // }

  static timing_data_class& get_instance() {
      static std::once_flag flag;
      // This will leak, but it's okay - it's the Trusty Leaky Singleton pattern.
      static timing_data_class* instance;
      std::call_once(flag, []() {
          instance = new timing_data_class();
      });
      return *instance;
  }

  /**
   * Add a new entry to the timing data.
   * @param name The name to be associated with the entry.  When called by a
   * timer this should be the name of the timer.
   * @param time Duration to be associated with the entry.  When called by a
   * timer this should be the elapsed time since the timer was started,
   * in chrono::duration format.
   */
  void insert_entry(const std::string& name, const duration_type& time) {
    std::lock_guard<std::mutex> lock(mtx_);
    interval_times_.insert(std::make_pair(name, time));
  }

  /**
   * Return a vector of the individual times logged with a given name.
   * @tparam D Duration type specifying the units associated with the returned
   * values.
   * @param string Name of the timer to be queried.
   * @return Vector of the individual times logged with the given name.
   */
  template <class D = std::chrono::milliseconds>
  auto get_entries_separately(const std::string& string) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<double> intervals;

    auto range = interval_times_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      intervals.push_back((std::chrono::duration_cast<D>(i->second)).count());
    }
    return intervals;
  }

  /**
   * Return the sum of the individual times logged with a given name.
   * @tparam D Duration type specifying the units associated with the returned
   * values.
   * @param string Name of the timer to be queried.
   * @return Cumulative duration of all the durations logged with the given
   * name.
   */
  template <class D = std::chrono::milliseconds>
  auto get_entries_summed(const std::string& string) const {
    std::lock_guard<std::mutex> lock(mtx_);
    double sum = 0.0;
    auto range = interval_times_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      sum += (std::chrono::duration_cast<D>(i->second)).count();
    }
    return sum;
  }

  /**
   * Return a vector of the names of all timers that have logged data.
   * @return Vector of the names of all timers that have logged data.
   */
  auto get_timer_names() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::set<std::string> multinames;

    std::vector<std::string> names;

    for (const auto& i : interval_times_) {
      multinames.insert(i.first);
    }
    names.reserve(multinames.size());
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
};

inline timing_data_class& get_timing_data_instance() {
  return timing_data_class::get_instance();
}

static timing_data_class& _timing_data{get_timing_data_instance()};

/**
 * Timer class for logging timing data.  Internnally aintains a start time and a
 * stop time (which are std::chrono::time_points).  The constructor, `start()`,
 * and `stop()` methods control operation and logging of the timer.
 */
class log_timer {
 private:
  using time_t = timing_data_class::time_type;
  using clock_t = timing_data_class::clock_type;
  time_t start_time, stop_time;
  std::string msg_;
  bool noisy_{false};

 public:
  /**
   * Constructor.  Associates a name with the timer and records the start time.
   * If the noisy flag is enabled, the timer will optionally prints a message.
   * @param msg The name to be associated with the timer.
   * @param noisy Flag to enable/disable printing of messages.
   */
  explicit log_timer(const std::string& msg = "unknown", bool noisy = false)
      : start_time(clock_t::now())
      , stop_time(start_time)
      , msg_(msg)
      , noisy_(noisy) {
    noisy_ |= _timing_data.get_verbose();
    if (noisy_) {
      std::cout << "# Starting timer " << msg_ << std::endl;
    }
  }

  /**
   * Start the timer.  Records the start time and optionally prints a message.
   * @return The start time.
   */
  time_t start() {
    if (noisy_) {
      std::cout << "# Starting timer " << msg_ << std::endl;
    }
    return (start_time = clock_t::now());
  }

  /**
   * Stop the timer.  Records the stop time and optionally prints a message.
   * @return The stop time.
   */
  time_t stop() {
    stop_time = clock_t::now();
    _timing_data.insert_entry(msg_, stop_time - start_time);

    if (noisy_) {
      std::cout << "# Stopping timer " << msg_ << ": "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       stop_time - start_time)
                       .count()
                << " ms" << std::endl;
    }
    return stop_time;
  }

  /**
   * Return the name associated with the timer.
   * @return The name associated with the timer.
   */
  const std::string& name() const {
    return msg_;
  }
};

/**
 * Scoped timer class for logging timing data.  Internnally maintains a start
 * time and a stop time (which are std::chrono::time_points).  The constructor
 * and destructor control operation and logging of the timer. It inherits from
 * `log_timer` but invokes `stop()` in its destructor. It is intended to measure
 * the lifetime of a scope.  It begins timing when it is constructed and stops
 * timing when it is destructed.
 */
class scoped_timer : public log_timer {
 public:
  explicit scoped_timer(const std::string& msg = "", bool noisy = false)
      : log_timer(msg, noisy) {
  }

  /**
   * Stop the timer when it goes out of scope.
   */
  ~scoped_timer() {
    this->stop();
  }
};

/**
 * Singleton class for recording memory consumption.  Internally maintains a
 * multimap of names and memory consumption values.  The `insert_entry()` method
 * is used to record memory consumption.  The `get_entries_*()` methods are used
 * to query the recorded memory consumption.
 *
 * Data are stored as bytes.  The `get_entries_*()` methods return values
 * as MiB (2^10 bytes).
 */
class memory_data {
 public:
  using memory_type = size_t;

 private:
  std::multimap<std::string, memory_type> memory_usages_;
  mutable std::mutex mtx_;
  bool verbose_{false};

  /**
   * Constructor.  Private to enforce singleton pattern.
   */
  memory_data() = default;
  ~memory_data() = default;

 public:
  /**
   * Copy constructor.  Deleted to enforce singleton pattern.
   */
  memory_data(const memory_data&) = delete;
  memory_data& operator=(const memory_data&) = delete;

  /**
   * Get a reference to the singleton instance of the class.
   * @return Reference to the singleton instance of the class.
   */
  // static memory_data& get_instance() {
  //   // std::cout << "[memory_data@get_instance]" << std::endl;
  //   static memory_data instance;
  //   return instance;
  // }

  static memory_data& get_instance() {
    static std::once_flag flag;
    // This will leak, but it's okay - it's the Trusty Leaky Singleton pattern.
    static memory_data* instance;
    std::call_once(flag, []() {
        instance = new memory_data();
    });
    return *instance;
  }

  // static memory_data& get_instance() {
  //   std::cout << "[memory_data@get_instance]" << std::endl;
  //     static std::once_flag flag;
  //     static std::unique_ptr<memory_data> instance;
  //     std::call_once(flag, []() {
  //         instance.reset(new memory_data());
  //     });
  //     std::cout << "[memory_data@get_instance] instance: " << instance << std::endl;
  //     return *instance;
  // }


  /**
   * Insert a memory consumption entry into the multimap.
   * @param name The name to be associated with the memory consumption.
   * @param use The memory consumption to be recorded (in bytes).
   */
  void insert_entry(const std::string& name, const memory_type& use) {
    std::lock_guard<std::mutex> lock(mtx_);
    memory_usages_.insert(std::make_pair(name, use));
  }

  /**
   * Get the memory consumption entries associated with a name.
   * @param string Name to be queried.
   * @return Vector of memory consumption values associated with the name.
   */
  auto get_entries_separately(const std::string& string) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<double> usages;

    auto range = memory_usages_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      usages.push_back(i->second / (1024 * 1024));
    }
    return usages;
  }

  /**
   * Get the sum of the memory consumption entries associated with a name.
   * @param string Name to be queried.
   * @return Vector of memory consumption values associated with the name.
   */
  auto get_entries_summed(const std::string& string) const {
    std::lock_guard<std::mutex> lock(mtx_);
    double sum = 0.0;
    auto range = memory_usages_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      sum += i->second;
    }
    return sum / (1024 * 1024);
  }

  /**
   * Get the names associated with the memory consumption entries.
   * @return Vector of names associated with the memory consumption entries.
   */
  auto get_usage_names() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::set<std::string> multinames;

    std::vector<std::string> names;

    for (const auto& i : memory_usages_) {
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
};

inline memory_data& get_memory_data_instance() {
  return memory_data::get_instance();
}

static memory_data& _memory_data{get_memory_data_instance()};

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
  // static count_data& get_instance() {
  //   static count_data instance;
  //   return instance;
  // }

  static count_data& get_instance() {
      std::cout << "[count_data@get_instance]" << std::endl;
      static std::once_flag flag;
      // This will leak, but it's okay - it's the Trusty Leaky Singleton pattern.
      static count_data* instance;
      std::call_once(flag, []() {
          instance = new count_data();
      });
      return *instance;
  }

  // static count_data& get_instance() {
  //     static std::once_flag flag;
  //     static std::unique_ptr<count_data> instance;
  //     std::call_once(flag, []() {
  //         instance.reset(new count_data());
  //     });
  //     return *instance;
  // }

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

#endif  // TDB_LOGGING_H
