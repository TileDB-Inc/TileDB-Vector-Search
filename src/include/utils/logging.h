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
#include <set>
#include <string>

#include "nlohmann/json.hpp"

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

using json = nlohmann::json;

class timing_data_class {
 public:
  using clock_type = std::chrono::high_resolution_clock;
  using time_type = std::chrono::time_point<clock_type>;
  using duration_type =
      std::chrono::duration<clock_type::rep, clock_type::period>;
  using name_time = std::multimap<std::string, duration_type>;

 private:
  name_time interval_times_;
  bool verbose_{false};
  bool debug_{false};

  timing_data_class() = default;
  ~timing_data_class() = default;

 public:
  timing_data_class(const timing_data_class&) = delete;
  timing_data_class& operator=(const timing_data_class&) = delete;

  static timing_data_class& get_instance() {
    static timing_data_class instance;
    return instance;
  }

  void insert_entry(const std::string& name, const duration_type& time) {
    interval_times_.insert(std::make_pair(name, time));
  }

  template <class D = std::chrono::milliseconds>
  auto get_entries_separately(const std::string& string) {
    std::vector<double> intervals;

    auto range = interval_times_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      intervals.push_back((std::chrono::duration_cast<D>(i->second)).count());
    }
    return intervals;
  }

  template <class D = std::chrono::milliseconds>
  auto get_entries_summed(const std::string& string) {
    double sum = 0.0;
    auto range = interval_times_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      sum += (std::chrono::duration_cast<D>(i->second)).count();
    }
    return sum;
  }

  auto get_timer_names() {
    std::set<std::string> multinames;

    std::vector<std::string> names;

    for (auto& i : interval_times_) {
      multinames.insert(i.first);
    }
    for (auto& i : multinames) {
      names.push_back(i);
    }
    return names;
  }

  void set_verbose(bool verbose) {
    verbose_ = verbose;
  }

  bool get_verbose() {
    return verbose_;
  }

  void set_debug(bool debug) {
    debug_ = debug;
  }

  bool get_debug() {
    return debug_;
  }
};

inline timing_data_class& get_timing_data_instance() {
  return timing_data_class::get_instance();
}

timing_data_class& _timing_data{get_timing_data_instance()};

class log_timer {
 private:
  using time_t = timing_data_class::time_type;
  using clock_t = timing_data_class::clock_type;
  time_t start_time, stop_time;
  std::string msg_;
  bool noisy_ {false};

 public:
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

  time_t start() {
    if (noisy_) {
      std::cout << "# Starting timer " << msg_ << std::endl;
    }
    return (start_time = clock_t::now());
  }

  time_t stop() {
    stop_time = clock_t::now();
    _timing_data.insert_entry(msg_, stop_time - start_time);

    if (noisy_) {
      std::cout << "# Stopping timer " << msg_ << ": " <<
          std::chrono::duration_cast<std::chrono::milliseconds>(stop_time-start_time).count() << " ms" << std::endl;
    }
    return stop_time;
  }

  std::string name() const {
    return msg_;
  }
};


class scoped_timer : public log_timer {
 public:
  explicit scoped_timer(const std::string& msg = "", bool noisy = false)
      : log_timer(msg, noisy) {
  }

  ~scoped_timer() {
    this->stop();
  }
};


class memory_data {
 public:
  using memory_type = size_t;
  using name_memory = std::multimap<std::string, memory_type>;

 private:
  name_memory memory_usages_;
  bool verbose_{false};
  bool debug_{false};

  memory_data() = default;
  ~memory_data() = default;

 public:
  memory_data(const memory_data&) = delete;
  memory_data& operator=(const memory_data&) = delete;

  static memory_data& get_instance() {
    static memory_data instance;
    return instance;
  }

  void insert_entry(const std::string& name, const memory_type& time) {
    memory_usages_.insert(std::make_pair(name, time));
  }

  template <class D = std::chrono::milliseconds>
  auto get_entries_separately(const std::string& string) {
    std::vector<double> usages;

    auto range = memory_usages_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      usages.push_back((std::chrono::duration_cast<D>(i->second)).count());
    }
    return usages;
  }

  auto get_entries_summed(const std::string& string) {
    double sum = 0.0;
    auto range = memory_usages_.equal_range(string);
    for (auto i = range.first; i != range.second; ++i) {
      sum += i->second;
    }
    return sum / (1024*1024);
  }

  auto get_usage_names() {
    std::set<std::string> multinames;

    std::vector<std::string> names;

    for (auto& i : memory_usages_) {
      multinames.insert(i.first);
    }
    for (auto& i : multinames) {
      names.push_back(i);
    }
    return names;
  }

  void set_verbose(bool verbose) {
    verbose_ = verbose;
  }

  bool get_verbose() {
    return verbose_;
  }

  void set_debug(bool debug) {
    debug_ = debug;
  }

  bool get_debug() {
    return debug_;
  }
};

inline memory_data& get_memory_data_instance() {
  return memory_data::get_instance();
}

memory_data& _memory_data{get_memory_data_instance()};


class timing_data {
 private:
  json timings;
  json metadata;
  json search_data;

  timing_data() = default;
  ~timing_data() = default;

 public:
  timing_data(const timing_data&) = delete;
  timing_data& operator=(const timing_data&) = delete;

  static timing_data& get_instance() {
    static timing_data instance;
    return instance;
  }

  // @todo update this to record per-invocation times as well as
  // cumulative times for same timer
  void add_timing(const std::string& operation, double elapsedTime) {
    timings[operation] = elapsedTime;
  }

  auto get_timings() {
    return timings;
  }

  void add_metadata(const std::string& key, const std::string& value) {
    metadata[key] = value;
  }

  auto get_metadata() {
    return metadata;
  }

  template <typename T>
  void add_search_datum(const std::string& key, T value) {
    search_data[key] = value;
  }
};

inline timing_data& get_json_timing_data_instance() {
  return timing_data::get_instance();
}

/*
 * Can also use this pattern:
 *
 * timing_data& get_timing_data_instance() {
 *   static timing_data instance;
 *   return instance;
 * }
 */

timing_data& _json_timing_data{get_json_timing_data_instance()};

void add_timing(const std::string& operation, double elapsedTime) {
  _json_timing_data.add_timing(operation, elapsedTime);
}

auto get_timings() {
  return _json_timing_data.get_timings();
}

#endif  // TDB_LOGGING_H
