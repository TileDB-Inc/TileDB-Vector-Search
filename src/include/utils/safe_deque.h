/**
 * @file   utils/safe_deque.h
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
 * A thread-safe deque implementation.
 *
 */

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>


/**
 * A thread-safe deque implementation.
 * @tparam T The type of element to store in the deque.
 */
template <class T>
class safe_deque {
 public:

  /**
   * Pop an element from the deque. If the deque is empty, this will block until
   * an element is available or until the deque is finished.
   * @return The popped element.
   */
  std::optional<T> pop() {
    std::unique_lock lock(mutex_);

    cv_.wait(lock, [this]() { return done_ || (deque_.empty() == false); });

    if (deque_.empty() == true || done_) {
      return {};
    }

    auto item = deque_.front();
    deque_.pop_front();
    return item;
  }

  /**
   * Pop an element from the deque. If the deque is empty, this will return
   * immediately with an empty optional.
   * @return The popped element, if available, else empty.
   */
  std::optional<T> try_pop() {
    std::scoped_lock lock{mutex_};

    if (deque_.empty()) {
      return {};
    }
    auto item = deque_.front();
    deque_.pop_front();
    return item;
  }

  /**
   * Pop an element from the back of the deque. If the deque is empty, this will
   * return immediately with an empty optional.
   * @return
   */
  std::optional<T> try_pop_back() {
    std::scoped_lock lock{mutex_};

    if (deque_.empty()) {
      return {};
    }
    auto item = deque_.back();
    deque_.pop_back();
    return item;
  }

  /**
   * Push an element onto the deque. If the deque is finished, this will return
   * immediately without pushing the element.
   * @param item The item to push onto the deque.
   */
  void push(T item) {
    std::scoped_lock lock(mutex_);
    if (done_) {
      return;
    }
    deque_.push_front(std::move(item));
    max_size_ = std::max(max_size_, deque_.size());
    cv_.notify_one();
  }

  /**
   * Perform a soft shutdown of the queue.  New elements cannot be added, but
   * existing elements will be processed.
   */
  void finish() {
    std::unique_lock lock(mutex_);
    done_ = true;
    cv_.notify_all();
  }

  // Some useful functions -- but not thread safe
  auto unsafe_max_size() {
    return max_size_;
  }
  auto unsafe_size() {
    return deque_.size();
  }
  auto unsafe_begin() {
    return deque_.begin();
  }
  auto unsafe_end() {
    return deque_.end();
  }

 private:
  std::deque<T> deque_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool done_{false};
  size_t max_size_{0};
};
