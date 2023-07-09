//
// This file is part of the course materials for CSE P 524 at the University of Washington,
// Winter 2022
//
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// Author: Andrew Lumsdaine
//

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

template <class T>
class safe_deque {

public:
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

  std::optional<T> try_pop() {
    std::scoped_lock lock{mutex_};

    if (deque_.empty()) {
      return {};
    }
    auto item = deque_.front();
    deque_.pop_front();
    return item;
  }

  std::optional<T> try_pop_back() {
    std::scoped_lock lock{mutex_};

    if (deque_.empty()) {
      return {};
    }
    auto item = deque_.back();
    deque_.pop_back();
    return item;
  }

  void push(T item) {
    std::scoped_lock lock(mutex_);
    if (done_) {
      return;
    }
    deque_.push_front(std::move(item));
    max_size_ = std::max(max_size_, deque_.size());
    cv_.notify_one();
  }

  void finish() {
    std::unique_lock lock(mutex_);
    done_ = true;
    cv_.notify_all();
  }

  // Some useful functions -- but not thread safe
  auto unsafe_max_size() { return max_size_; }
  auto unsafe_size() { return deque_.size(); }
  auto unsafe_begin() { return deque_.begin(); }
  auto unsafe_end() { return deque_.end(); }

private:
  std::deque<T>           deque_;
  mutable std::mutex      mutex_;
  std::condition_variable cv_;
  bool                    done_{false};
  size_t                  max_size_{0};
};
