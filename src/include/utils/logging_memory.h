/**
 * @file   logging_memory.h
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

#ifndef TDB_LOGGING_MEMORY_H
#define TDB_LOGGING_MEMORY_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

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
  static memory_data& get_instance() {
    static std::once_flag flag;
    // This will leak, but it's okay - it's the Trusty Leaky Singleton pattern.
    static memory_data* instance;
    std::call_once(flag, []() { instance = new memory_data(); });
    return *instance;
  }

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

  std::string dump() const {
    std::ostringstream oss;
    auto usage_names = get_usage_names();
    for (const auto& name : usage_names) {
      auto usages = get_entries_separately(name);
      double sum_mib = get_entries_summed(name);
      double average_mib = sum_mib / usages.size();

      oss << name << ":";
      oss << " count: " << usages.size() << ",";
      oss << " sum: " << format_memory_mib(sum_mib) << ",";
      oss << " avg: " << format_memory_mib(average_mib);
      oss << "\n";
    }
    return oss.str();
  }

 private:
  std::string format_memory_mib(double memory_mib) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (memory_mib < 1) {
      oss << (memory_mib * 1024);
      return oss.str() + " KiB";
    } else if (memory_mib < 1024) {
      oss << memory_mib;
      return oss.str() + " MiB";
    } else {
      oss << (memory_mib / 1024);
      return oss.str() + " GiB";
    }
  }
};

inline memory_data& get_memory_data_instance() {
  return memory_data::get_instance();
}

static memory_data& _memory_data{get_memory_data_instance()};

#endif  // TDB_LOGGING_MEMORY_H
