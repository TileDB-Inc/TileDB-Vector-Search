/**
 * @file   utils/threadpool.h
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
 * A work-stealing threadpool implementation.
 *
 */

#ifndef TILEDB_THREADPOOL_H
#define TILEDB_THREADPOOL_H

#include "safe_deque.h"

#include <functional>
#include <future>
#include <iostream>
#include <type_traits>
#include <vector>

namespace impl {

// #define WORK_STEALING
#define MULTI
// #define NO_RECURSIVE_PUSH

class threadpool {
 private:
  threadpool() = delete;

  ~threadpool() {
#ifdef MULTI
    for (auto& t : task_queues_) {
      t.finish();
    }
#else
    task_queue_.finish();
#endif
    for (auto&& t : threads_) {
      t.join();
    }
    threads_.clear();
  }

 private:
  explicit threadpool(size_t concurrency = std::thread::hardware_concurrency())
      : num_threads_(concurrency)
      ,
#ifdef MULTI
      task_queues_(num_threads_)
#else
      task_queue_()
#endif
  {

    threads_.reserve(num_threads_);

    for (size_t i = 0; i < num_threads_; ++i) {
      std::thread tmp = std::thread(&threadpool::worker, this, i);
      threads_.emplace_back(std::move(tmp));
    }
    // std::cout << "concurrency: " << threads_.size() << std::endl;
  }

 public:
  /**
   * Delete copy constructor and assignment operator for singleton.
   */
  threadpool(const threadpool&) = delete;
  threadpool& operator=(const threadpool&) = delete;

  /**
   * Return a reference to the singleton instance.
   * @return The singleton instance.
   */
  static threadpool& get_instance(
      size_t concurrency = std::thread::hardware_concurrency()) {
    static threadpool instance{concurrency};
    return instance;
  }

 public:

  /**
   * @brief Submit a task to the threadpool.
   * @tparam Fn The type of the task.
   * @tparam Args The types of the arguments to the task.
   * @param f The function to execute.
   * @param args The arguments to the function.
   * @return A future to the result of the task.
   */
  template <class Fn, class... Args>
  auto async(Fn&& f, Args&&... args) {
    using R = std::invoke_result_t<std::decay_t<Fn>, std::decay_t<Args>...>;

    std::shared_ptr<std::promise<R>> task_promise(new std::promise<R>);
    std::future<R> future = task_promise->get_future();

    auto task = std::make_shared<std::function<void()>>(
        [f = std::forward<Fn>(f),
         args = std::make_tuple(std::forward<Args>(args)...),
         task_promise]() mutable {
          try {
            if constexpr (std::is_void_v<R>) {
              std::apply(std::move(f), std::move(args));
              task_promise->set_value();
            } else {
              task_promise->set_value(
                  std::apply(std::move(f), std::move(args)));
            }
          } catch (...) {
            task_promise->set_exception(std::current_exception());
          }
        });

    // std::thread(*task).detach();

#ifdef NO_RECURSIVE_PUSH
    bool found = false;
    for (auto& j : threads_) {
      if (j.get_id() == std::this_thread::get_id()) {
        //	std::cout << "found" << std::endl;
        found = true;
        break;
      }
    }

    if (found) {
      (*task)();
    } else {
#ifdef MULTI
      size_t i = index_++;
      task_queues_[i % num_threads_].push(task);
#else
      task_queue_.push(task);
#endif
    }
#else
#ifdef MULTI
    size_t i = index_++;
    task_queues_[i % num_threads_].push(task);
#else
    task_queue_.push(task);
#endif
#endif

    return future;
  }

#ifdef WORK_STEALING
  template <class R>
  auto wait(std::future<R>&& task) {
    while (true) {
      if (task.wait_for(std::chrono::milliseconds(0)) ==
          std::future_status::ready) {
        if constexpr (std::is_void_v<R>) {
          task.wait();
          return;
        } else {
          auto ret = task.get();
          return ret;
        }
      } else {
        std::optional<std::shared_ptr<std::function<void()>>> val;

#ifdef MULTI
        size_t i = index_++;
        for (size_t j = 0; j < num_threads_ * rounds_; ++j) {
          val = task_queues_[(i + j) % num_threads_].try_pop();

          if (val) {
            break;
          }
        }
#else
        val = task_queue_.try_pop();
#endif

        if (val) {
          (*(*val))();
        } else {
          // task.wait_for(std::chrono::milliseconds(3));
          std::this_thread::yield();
        }
      }
    }
  }

  template <class R>
  auto wait(std::future<R>& task) {
    return wait(std::move(task));
  }
#endif

  auto num_threads() {
    return num_threads_;
  }

  auto get_thread_id() {
    return thread_id;
  }

 private:
  /**
   * Class variable to hold the id of the thread, in the range [0, num_threads_).
   */
  static thread_local size_t thread_id;

  /**
   * The worker function for each thread.  The function loops, trying to get
   * tasks from the queue.  If the queue is empty, the thread will try to steal
   * tasks from other threads. If the thread cannot steal any tasks, it will
   * block on its own queue.
   * @param i Identifier for the thread, in the range [0, num_threads_).
   */
  void worker(size_t i) {
    thread_id = i;
    while (true) {
      std::optional<std::shared_ptr<std::function<void()>>> val;

#ifdef MULTI
      val = task_queues_[i].try_pop();
      if (val) {
        (*(*val))();
      } else {
        for (size_t j = 0; j < num_threads_ * rounds_; ++j) {
          val = task_queues_[(i + j) % num_threads_].try_pop();
          if (val) {
            break;
          }
        }
        if (val) {
          (*(*val))();
        } else {
          val = task_queues_[i].pop();
          if (val) {
            (*(*val))();
          } else {
            break;
          }
        }
      }
#else
      val = task_queue_.pop();
      if (val) {
        (*(*val))();
      } else {
        break;
      }
#endif
    }
  }

  const size_t num_threads_;
  std::vector<std::thread> threads_;

#ifdef MULTI
  std::atomic<size_t> index_;
  const size_t rounds_{3};
  std::vector<safe_deque<std::shared_ptr<std::function<void()>>>> task_queues_;
#else
  safe_deque<std::shared_ptr<std::function<void()>>> task_queue_;
#endif
};

thread_local size_t threadpool::thread_id{0};

}  // namespace impl

auto& threadpool = impl::threadpool::get_instance();

void yack() {
  std::cout << threadpool.get_thread_id();
}

#endif  // TILEDB_THREADPOOL_H