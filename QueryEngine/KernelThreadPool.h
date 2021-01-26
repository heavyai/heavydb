/*
 * Copyright 2021 OmniSci, Inc.
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

#include <deque>
#include <future>
#include <memory>
#include <mutex>

#include "Logger/Logger.h"

#define OMNISCI_TASK 1

/**
 * A thread pool which supports thread re-use across tasks.  Tasks are submitted to the
 * queue. Notification is submitted to a worker thread for each submitted task to the
 * pool. An open thread will then pick up the task.
 *
 */
class KernelThreadPool {
 public:
#ifdef OMNISCI_TASK
  struct Task {
    Task(std::function<void()>&& f_in) : f(std::move(f_in)) {}

    std::future<void> get_future() { return promise.get_future(); }

    void operator()() {
      try {
        f();
        promise.set_value();
      } catch (...) {
        try {
          promise.set_exception(std::current_exception());
        } catch (std::exception& e) {
          LOG(FATAL) << e.what();
        }
      }
    }

    std::function<void()> f;
    std::promise<void> promise;
  };
#else
  using Task = std::packaged_task<void()>;
#endif

  KernelThreadPool(const size_t num_hardware_threads)
      : workers_(num_hardware_threads), per_thread_tasks_(num_hardware_threads) {
    CHECK_EQ(workers_.size(), per_thread_tasks_.size());
    for (size_t i = 0; i < workers_.size(); i++) {
      workers_[i] = std::thread(&KernelThreadPool::worker, this, i);
    }
  }

  ~KernelThreadPool() {
    {
      std::lock_guard<decltype(ready_mutex_)> lock(ready_mutex_);
      threads_should_exit_ = true;
    }
    ready_cv_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  std::future<void> submit(Task&& task) {
    std::unique_lock ready_lock(ready_mutex_);
    auto future = task.get_future();
    main_tasks_.push_back(std::move(task));
    ready_lock.unlock();
    ready_cv_.notify_one();
    return future;
  }

  std::vector<std::future<void>> submitBatch(std::vector<Task>&& tasks) {
    std::unique_lock ready_lock(ready_mutex_);
    const size_t num_tasks = tasks.size();
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < num_tasks; i++) {
      main_tasks_.emplace_back(std::move(tasks[i]));
      futures.emplace_back(main_tasks_.back().get_future());
    }
    ready_lock.unlock();
    for (size_t i = 0; i < num_tasks; i++) {
      if (i == workers_.size()) {
        break;
      }
      ready_cv_.notify_one();
    }
    return futures;
  }

  void clearQueue() noexcept {
    std::lock_guard<decltype(ready_mutex_)> lock(ready_mutex_);
    main_tasks_.clear();
  }

  size_t numWorkers() const { return workers_.size(); }

 private:
  void worker(const size_t thread_idx) {
    std::unique_lock<std::mutex> ready_lock(ready_mutex_);
    while (true) {
      ready_cv_.wait(ready_lock,
                     [this] { return !main_tasks_.empty() || threads_should_exit_; });

      if (threads_should_exit_) {
        return;
      }

      CHECK_LT(thread_idx, per_thread_tasks_.size());
      auto& this_threads_tasks = per_thread_tasks_[thread_idx];

      // on wake up, greedily take tasks from the queue
      // by default we will take the first N tasks / M workers tasks.
      // we have exclusive access to the main task queue, so this should be safe
      std::unique_lock<std::mutex> thread_lock(this_threads_tasks.thread_queue_mutex);
      // grab at least one task
      this_threads_tasks.thread_queue.push_back(std::move(main_tasks_.front()));
      main_tasks_.pop_front();

      // get additional tasks
      auto num_tasks_to_steal = ceil(main_tasks_.size() / workers_.size());
      for (size_t i = 0; i < num_tasks_to_steal; i++) {
        if (main_tasks_.empty()) {
          break;
        }
        this_threads_tasks.thread_queue.push_back(std::move(main_tasks_.front()));
        main_tasks_.pop_front();
      }

      VLOG(1) << "Thread " << thread_idx << " assigned "
              << this_threads_tasks.thread_queue.size() << " tasks.";
      // allow other threads to pick up tasks
      ready_lock.unlock();

      auto current_task = std::move(this_threads_tasks.thread_queue.front());
      this_threads_tasks.thread_queue.pop_front();
      thread_lock.unlock();

      while (true) {
        current_task();
        thread_lock.lock();
        if (!this_threads_tasks.thread_queue.empty()) {
          current_task = std::move(this_threads_tasks.thread_queue.front());
          this_threads_tasks.thread_queue.pop_front();
        } else {
          break;
        }
        thread_lock.unlock();
      }

      VLOG(1) << "Thread " << thread_idx << " finished.";
      // wait for signal
      ready_lock.lock();
    }
  }

  std::vector<std::thread> workers_;

  // ready mutex and ready cv guard current_task_idx and main_tasks_
  std::mutex ready_mutex_;
  std::condition_variable ready_cv_;
  std::deque<Task> main_tasks_;  // protected shared variable
  struct ProtectedThreadQueue {
    ProtectedThreadQueue() {}

    std::deque<Task> thread_queue;
    std::mutex thread_queue_mutex;
  };
  std::vector<ProtectedThreadQueue> per_thread_tasks_;
  bool threads_should_exit_{false};
};
