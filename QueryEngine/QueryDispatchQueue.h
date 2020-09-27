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
  using Task = std::packaged_task<void(size_t)>;

  QueryDispatchQueue(const size_t parallel_executors_max) {
    workers_.resize(parallel_executors_max);
    for (size_t i = 0; i < workers_.size(); i++) {
      // worker IDs are 1-indexed, leaving Executor 0 for non-dispatch queue worker tasks
      workers_[i] = std::thread(&QueryDispatchQueue::worker, this, i + 1);
    }
  }

  /**
   * Submit a new task to the queue. Blocks until the task begins execution. The caller is
   * expected to maintain a copy of the shared_ptr which will be used to access results
   * once the task runs.
   */
  void submit(std::shared_ptr<Task> task, const bool is_update_delete) {
    if (workers_.size() == 1 && is_update_delete) {
      std::lock_guard<decltype(update_delete_mutex_)> update_delete_lock(
          update_delete_mutex_);
      CHECK(task);
      // We only have 1 worker. Run this task on the calling thread on a special, second
      // worker. The task is under the update delete lock, so we don't have to worry about
      // contention on the special worker. This protects against deadlocks should the
      // query running (or any pending queries) hold a read lock on something that
      // requires a write lock during update/delete.
      (*task)(2);
      return;
    }
    std::unique_lock<decltype(queue_mutex_)> lock(queue_mutex_);

    LOG(INFO) << "Dispatching query with " << queue_.size() << " queries in the queue.";
    queue_.push(task);
    lock.unlock();
    cv_.notify_all();
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
        auto task = queue_.front();
        queue_.pop();

        LOG(INFO) << "Worker " << worker_idx
                  << " running query and returning control. There are now "
                  << queue_.size() << " queries in the queue.";
        // allow other threads to pick up tasks
        lock.unlock();
        CHECK(task);
        (*task)(worker_idx);

        // wait for signal
        lock.lock();
      }
    }
  }

  std::mutex queue_mutex_;
  std::condition_variable cv_;

  std::mutex update_delete_mutex_;

  bool threads_should_exit_{false};
  std::queue<std::shared_ptr<Task>> queue_;
  std::vector<std::thread> workers_;
};
