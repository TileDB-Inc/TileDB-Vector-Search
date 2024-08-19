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

static constexpr const char* ROOT_NAME = "root";

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
 * Singleton class for storing timing data in a tree structure.
 *
 * As a singleton, the object from this class can be used from anywhere in the
 * code. Timing data is stored in a tree structure, where each node represents a
 * timer, storing only the total duration and the count of occurrences.
 * Each thread maintains its own separate tree to avoid conflicts in
 * multithreaded scenarios.
 */
class timing_data_class {
 public:
  using clock_type = std::chrono::high_resolution_clock;
  using time_type = std::chrono::time_point<clock_type>;
  using duration_type =
      std::chrono::duration<clock_type::rep, clock_type::period>;

  struct TimerNode {
    std::string name;
    duration_type total_duration;
    int count;
    std::vector<std::unique_ptr<TimerNode>> children;

    TimerNode(const std::string& name)
        : name(name)
        , total_duration(duration_type::zero())
        , count(0) {
    }

    std::string dump() const {
      return "(name: " + name + ", count: " + std::to_string(count) + ", total_duration: " + std::to_string(total_duration.count()) + ", address: " + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ")";
    }
  };

 private:
  // Thread-local storage for maintaining a separate timing tree per thread.
  static thread_local std::unique_ptr<TimerNode> root_;
  static thread_local TimerNode* current_node_;

  // Global storage for all thread trees. Used to dump() the timing data from
  // all threads.
  std::vector<TimerNode*> all_roots_;
  mutable std::mutex mutex_;

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
  static timing_data_class& get_instance() {
    static std::once_flag flag;
    // This will leak, but it's okay - it's the Trusty Leaky Singleton pattern.
    static timing_data_class* instance;
    std::call_once(flag, []() { instance = new timing_data_class(); });
    return *instance;
  }

  /**
   * Start a new timer node as a child of the current node, or reuse an existing
   * one.
   * @param name The name of the timer.
   * @return Pointer to the newly created or existing timer node.
   */
  void start_timer(const std::string& name) {
    if (!root_) {
      std::cout << "[logging@start_timer] Creating root node" << std::endl;
      root_ = std::make_unique<TimerNode>("root");
      std::cout << "[logging@start_timer] root_: " << root_->dump() << std::endl;
      current_node_ = root_.get();
      std::cout << "[logging@start_timer] current_node_: " << current_node_->dump() << std::endl;

      // Save the root so we can dump its timing data later.
      std::lock_guard<std::mutex> lock(mutex_);
      all_roots_.push_back(root_.get());
    }

    // Check if a child with the same name already exists
    for (auto& child : current_node_->children) {
      if (child->name == name) {
        current_node_ = child.get();
        std::cout << "[logging@start_timer] Existing child node" << std::endl;
        return;
      }
    }

    // If no existing child, create a new one
    std::cout << "[logging@start_timer] Creating child node" << std::endl;
    auto new_node = std::make_unique<TimerNode>(name);
    TimerNode* node_ptr = new_node.get();
    current_node_->children.push_back(std::move(new_node));
    current_node_ = node_ptr;
  }

  /**
   * Stop the current timer and move back to the parent node.
   * @param start_time The start time of the timer to be stopped.
   */
  void stop_timer(const time_type& start_time) {
    if (!current_node_ || !root_) {
      return;
    }
    std::cout << "[logging@stop_timer] root_: " << root_->dump() << std::endl;
    auto end_time = clock_type::now();

    // std::cout << "[logging@stop_timer] current_node_: " << (current_node_ == nullptr ? "null" : "okay") << std::endl;
    current_node_->total_duration += (end_time - start_time);
    current_node_->count += 1;
    std::cout << "[logging@stop_timer] current_node_: " << current_node_->dump() << std::endl;

    if (current_node_ != root_.get()) {
      current_node_ = find_parent(root_.get(), current_node_);
    }
  }

  void set_verbose(bool verbose) {
    verbose_ = verbose;
  }

  bool get_verbose() const {
    return verbose_;
  }

  /**
   * Dump the timing data for all threads in a hierarchical format, printing the
   * average duration.
   * @return String representation of the timing data.
   */
  std::string dump() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    std::cout << "[logging@dump] all_roots_.size(): " << all_roots_.size() << std::endl;
    for (size_t i = 0; i < all_roots_.size(); ++i) {
      oss << "Thread " << i + 1 << " Timing Data:\n";
      std::cout << "[logging@dump] all_roots_[i]" << all_roots_[i]->dump() << std::endl;
      dump_recursive(oss, all_roots_[i], 0);
      oss << "\n";
    }

    return oss.str();
  }

 private:
  /**
   * Recursively find the parent of the given node.
   * @param root The current node in the search.
   * @param node The node for which to find the parent.
   * @return Pointer to the parent node.
   */
  TimerNode* find_parent(TimerNode* root, TimerNode* node) {
    for (const auto& child : root->children) {
      if (child.get() == node) {
        std::cout << "[logging@find_parent] Returning root" << root->dump() << std::endl;
        return root;
      } else {
        TimerNode* result = find_parent(child.get(), node);
        if (result)
        std::cout << "[logging@find_parent] Returning result" << result->dump() << std::endl;
          return result;
      }
    }
    std::cout << "[logging@find_parent] Returning nullptr" << std::endl;
    return nullptr;
  }

  /**
   * Helper function to recursively dump the timing data.
   * @param oss Output stream to write to.
   * @param node The current node being processed.
   * @param depth The depth of the current node in the tree.
   */
  void dump_recursive(
      std::ostringstream& oss, const TimerNode* node, int depth) const {
    std::cout << "[logging@dump_recursive] node: " << (node == nullptr ? "null" : node->name) << std::endl;
    std::string indent(depth * 2, ' ');
    double average_duration =
        (node->count > 0) ?
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                node->total_duration)
                    .count() /
                static_cast<double>(node->count) :
            0.0;
    if (node->name != ROOT_NAME) {
      oss << indent << node->name << ": count = " << node->count
          << ", avg = " << format_duration_ns(average_duration) << "\n";
    }
    for (const auto& child : node->children) {
      dump_recursive(oss, child.get(), depth + 1);
    }
  }

  std::string format_duration_ns(double duration_ns) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (duration_ns < 1000) {
      oss << duration_ns;
      return oss.str() + " ns";
    } else if (duration_ns < 1000000) {
      oss << (duration_ns / 1000.0);
      return oss.str() + " µs";
    } else if (duration_ns < 1000000000) {
      oss << (duration_ns / 1000000.0);
      return oss.str() + " ms";
    } else {
      oss << (duration_ns / 1000000000.0);
      return oss.str() + " s";
    }
  }
};

// Define thread-local storage for the timing tree. thread_local static members
// of a class must be defined outside the class definition
thread_local std::unique_ptr<timing_data_class::TimerNode>
    timing_data_class::root_ = nullptr;
thread_local timing_data_class::TimerNode* timing_data_class::current_node_ =
    nullptr;

inline timing_data_class& get_timing_data_instance() {
  return timing_data_class::get_instance();
}

static timing_data_class& _timing_data{get_timing_data_instance()};

/**
 * Timer class for logging timing data. Maintains a start time and a stop time.
 * The constructor, `start()`, and `stop()` methods control operation and
 * logging of the timer.
 */
class log_timer {
 private:
  using time_t = timing_data_class::time_type;
  using clock_t = timing_data_class::clock_type;
  time_t start_time;
  std::string msg_;
  bool noisy_{false};

 public:
  /**
   * Constructor. Associates a name with the timer and records the start time.
   * If the noisy flag is enabled, the timer will optionally print a message.
   * @param msg The name to be associated with the timer.
   * @param noisy Flag to enable/disable printing of messages.
   */
  explicit log_timer(const std::string& msg, bool noisy = false)
      : start_time(clock_t::now())
      , msg_(msg)
      , noisy_(noisy) {
    noisy_ |= _timing_data.get_verbose();
    if (noisy_) {
      std::cout << "# Starting timer " << msg_ << std::endl;
    }
    _timing_data.start_timer(msg_);
  }

  /**
   * Start the timer.  Records the start time and optionally prints a message.
   * @return The start time.
   */
  time_t start() {
    if (noisy_) {
      std::cout << "# Restarting timer " << msg_ << std::endl;
    }
    return (start_time = clock_t::now());
  }

  /**
   * Stop the timer.  Records the stop time and optionally prints a message.
   */
  void stop() {
    _timing_data.stop_timer(start_time);
    if (noisy_) {
      std::cout << "# Stopping timer " << msg_ << ": "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       clock_t::now() - start_time)
                       .count()
                << " ms" << std::endl;
    }
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
 * Scoped timer class for logging timing data. Maintains a start time and a stop
 * time. The constructor and destructor control operation and logging of the
 * timer. It inherits from `log_timer` but invokes `stop()` in its destructor.
 * It is intended to measure the lifetime of a scope.
 */
class scoped_timer : public log_timer {
 public:
  explicit scoped_timer(const std::string& msg, bool noisy = false)
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

#endif  // TDB_LOGGING_H
