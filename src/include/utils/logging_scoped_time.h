/**
 * @file logging_scoped_time.h
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
 * This file contains a scoped timer class for logging timing data. It is
 * thread-safe and will create a separate timing tree for each thread. Note
 * there we use a lock when timers start and stop, so you should not use this in
 * performance-critical code paths.
 */

#ifndef TDB_LOGGING_SCOPED_TIME_H
#define TDB_LOGGING_SCOPED_TIME_H

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
#include "utils/utils.h"

static constexpr const char* ROOT_NAME = "root";

class scoped_timing_data_class {
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
  scoped_timing_data_class() = default;
  ~scoped_timing_data_class() = default;

 public:
  /**
   * Delete copy constructor and assignment operator.
   */
  scoped_timing_data_class(const scoped_timing_data_class&) = delete;
  scoped_timing_data_class& operator=(const scoped_timing_data_class&) = delete;

  /**
   * Return a reference to the singleton instance.
   * @return The singleton instance.
   */
  static scoped_timing_data_class& get_instance() {
    static std::once_flag flag;
    // This will leak, but it's okay - it's the Trusty Leaky Singleton pattern.
    static scoped_timing_data_class* instance;
    std::call_once(flag, []() { instance = new scoped_timing_data_class(); });
    return *instance;
  }

  /**
   * Start a new timer node as a child of the current node, or reuse an existing
   * one.
   * @param name The name of the timer.
   */
  void start_timer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto thread_id = std::this_thread::get_id();
    if (threadToRoot_.find(thread_id) == threadToRoot_.end()) {
      threadToRoot_[thread_id] = std::make_unique<TimerNode>(ROOT_NAME);
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

  /**
   * Start a new timer node as a child of the current node, or reuse an existing
   * one.
   * @param start_time The time that this timer was started.
   */
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
      threadToCurrentNode_[thread_id] =
          find_parent(threadToRoot_[thread_id].get(), current_node);
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

  /**
   * Get the cumulative duration for a timer with the given name across all
   * threads.
   * @param name The name of the timer.
   * @return The summed duration in the specified units.
   */
  template <typename Duration = std::chrono::milliseconds>
  double get_entries_summed(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    double total_duration = 0.0;

    for (const auto& [thread_id, root] : threadToRoot_) {
      total_duration +=
          get_entries_summed_recursive<Duration>(root.get(), name);
    }

    return total_duration;
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
    if (node->name != ROOT_NAME) {
      oss << indent << node->name << ": count = " << node->count
          << ", avg = " << format_duration_ns(average_duration) << "\n";
    }
    for (const auto& child : node->children) {
      dump_recursive(oss, child.get(), depth + 1);
    }
  }

  template <typename Duration = std::chrono::milliseconds>
  double get_entries_summed_recursive(
      const TimerNode* node, const std::string& name) const {
    double duration_sum = 0.0;

    if (node->name == name) {
      duration_sum +=
          std::chrono::duration_cast<Duration>(node->total_duration).count();
    }

    for (const auto& child : node->children) {
      duration_sum += get_entries_summed_recursive<Duration>(child.get(), name);
    }

    return duration_sum;
  }
};

inline scoped_timing_data_class& get_scoped_timing_data_instance() {
  return scoped_timing_data_class::get_instance();
}

static scoped_timing_data_class& _scoped_timing_data{
    get_scoped_timing_data_instance()};

/**
 * Scoped timer class for logging timing data. Maintains a start time and a stop
 * time. The constructor and destructor control operation and logging of the
 * timer. It inherits from `log_timer` but invokes `stop()` in its destructor.
 * It is intended to measure the lifetime of a scope.
 */
class scoped_timer {
 private:
  using time_t = scoped_timing_data_class::time_type;
  using clock_t = scoped_timing_data_class::clock_type;
  time_t start_time_;
  std::string msg_;

 public:
  explicit scoped_timer(const std::string& msg)
      : start_time_(clock_t::now())
      , msg_(msg) {
    _scoped_timing_data.start_timer(msg_);
  }

  /**
   * Stop the timer when it goes out of scope.
   */
  ~scoped_timer() {
    _scoped_timing_data.stop_timer(start_time_);
  }
};

#endif  // TDB_LOGGING_SCOPED_TIME_H
