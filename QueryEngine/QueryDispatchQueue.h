/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

/**
 * QueryDispatchQueue maintains a list of pending queries and dispatches those queries as
 * Executors become available
 */
class QueryDispatchQueue {
 public:
  QueryDispatchQueue(const size_t parallel_executors_max) {
    workers_.resize(parallel_executors_max);
    for (size_t i = 0; i < workers_.size(); i++) {
      workers_[i] = std::thread(&QueryDispatchQueue::worker, this, i);
    }
  }

  /**
   * Submit a new task to the queue. Blocks until the task begins execution, and returns a
   * future to the executing task.
   */
  std::future<void> submit(std::packaged_task<void(size_t)>&& task) {
    std::unique_lock<decltype(queue_mutex_)> lock(queue_mutex_);

    LOG(INFO) << "Dispatching query with " << queue_.size() << " queries in the queue.";
    queue_.push(std::move(task));
    auto future = queue_.back().get_future();
    lock.unlock();
    cv_.notify_all();
    return future;
  }

  ~QueryDispatchQueue() {
    {
      std::lock_guard<decltype(queue_mutex_)> lock(queue_mutex_);
      threads_should_exit_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
  }

 private:
  void worker(const size_t worker_idx) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    while (true) {
      cv_.wait(lock, [this] { return !queue_.empty() || threads_should_exit_; });

      if (threads_should_exit_) {
        return;
      }

      if (!queue_.empty()) {
        auto task = std::move(queue_.front());
        queue_.pop();

        LOG(INFO) << "Running query and returning control. There are now "
                  << queue_.size() << " queries in the queue.";
        // allow other threads to pick up tasks
        lock.unlock();
        task(worker_idx);

        // wait for signal
        lock.lock();
      }
    }
  }

  std::mutex queue_mutex_;
  std::condition_variable cv_;

  bool threads_should_exit_{false};
  std::queue<std::packaged_task<void(size_t)>> queue_;
  std::vector<std::thread> workers_;
};
