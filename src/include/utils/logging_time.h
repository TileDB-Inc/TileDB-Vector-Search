/**
 * @file logging_time.h
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
      return "(name: " + name + ", count: " + std::to_string(count) +
             ", total_duration: " + std::to_string(total_duration.count()) +
             ", children: " + std::to_string(children.size()) + " address: " +
             std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ")";
    }
  };

 private:
  // Map for maintaining a separate timing tree per thread.
  std::unordered_map<std::thread::id, std::unique_ptr<TimerNode>> threadToRoot_;
  std::unordered_map<std::thread::id, TimerNode*> threadToCurrentNode_;

  mutable std::mutex mutex_;

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
    std::call_once(flag, []() {
      instance = new timing_data_class();
    });
    return *instance;
  }

  /**
   * Start a new timer node as a child of the current node, or reuse an existing
   * one.
   * @param name The name of the timer.
   * @return Pointer to the newly created or existing timer node.
   */
  void start_timer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto thread_id = std::this_thread::get_id();
    if (threadToRoot_.find(thread_id) == threadToRoot_.end()) {
      threadToRoot_[thread_id] = std::make_unique<TimerNode>("root");
      threadToCurrentNode_[thread_id] = threadToRoot_[thread_id].get();
    }
    auto current_node = threadToCurrentNode_[thread_id];
    for (auto& child : current_node->children) {
      if (child->name == name) {
        threadToCurrentNode_[thread_id] = child.get();
        return;
      }
    }

    current_node->children.push_back(std::make_unique<TimerNode>(name));
    threadToCurrentNode_[thread_id] = current_node->children.back().get();
  }

  void stop_timer(const time_type& start_time) {
    auto end_time = clock_type::now();
    std::lock_guard<std::mutex> lock(mutex_);
    auto thread_id = std::this_thread::get_id();
    if (threadToCurrentNode_.find(thread_id) == threadToCurrentNode_.end()) {
      return;
    }

    auto current_node = threadToCurrentNode_[thread_id];
    current_node->total_duration += (end_time - start_time);
    current_node->count += 1;

    if (current_node != threadToRoot_[thread_id].get()) {
      threadToCurrentNode_[thread_id] = find_parent(threadToRoot_[thread_id].get(), current_node);
    }
  }

  /**
   * Dump the timing data for all threads in a hierarchical format, printing the
   * average duration.
   * @return String representation of the timing data.
   */
  std::string dump() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    for (const auto& [thread_id, root] : threadToRoot_) {
      oss << "Thread " << thread_id << ":\n";
      dump_recursive(oss, root.get(), 0);
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
    if (!root || !node) {
      throw std::runtime_error("Invalid input");
    }
    if (root == node) {
      return root;
    }
    for (const auto& child : root->children) {
      if (child.get() == node) {
        return root;
      } else {
        TimerNode* result = find_parent(child.get(), node);
        if (result) {
          return result;
        }
      }
    }
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
    std::string indent(depth * 2, ' ');
    double average_duration =
        (node->count > 0) ?
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                node->total_duration)
                    .count() /
                static_cast<double>(node->count) :
            0.0;
    if (node->name != "root") {
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

 public:
  /**
   * Constructor. Associates a name with the timer and records the start time.
   * @param msg The name to be associated with the timer.
   */
  explicit log_timer(const std::string& msg)
      : start_time(clock_t::now())
      , msg_(msg){
    _timing_data.start_timer(msg_);
  }

  /**
   * Start the timer.  Records the start time and optionally prints a message.
   * @return The start time.
   */
  time_t start() {
    return (start_time = clock_t::now());
  }

  /**
   * Stop the timer.  Records the stop time and optionally prints a message.
   */
  void stop() {
    _timing_data.stop_timer(start_time);
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
  explicit scoped_timer(const std::string& msg)
      : log_timer(msg) {
  }

  /**
   * Stop the timer when it goes out of scope.
   */
  ~scoped_timer() {
    this->stop();
  }
};

#endif  // TDB_LOGGING_H
