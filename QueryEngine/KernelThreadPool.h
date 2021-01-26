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
#include <random>

#include "Logger/Logger.h"
#include "Shared/measure.h"

/**
 * A thread pool which supports thread re-use across tasks.  Tasks are submitted to the
 * queue. Notification is submitted to a worker thread for each submitted task to the
 * pool. An open thread will then pick up the task.
 *
 */
class KernelThreadPool {
 public:
  using Task = std::packaged_task<void()>;

  KernelThreadPool(const size_t num_hardware_threads)
      : threads_(num_hardware_threads), workers_(num_hardware_threads) {
    CHECK_EQ(threads_.size(), workers_.size());
    for (size_t i = 0; i < threads_.size(); i++) {
      threads_[i] = std::thread(&KernelThreadPool::worker, this, i);
    }
    dist_ = std::uniform_int_distribution<size_t>(0, workers_.size() - 1);
  }

  ~KernelThreadPool() {
    for (auto& worker : workers_) {
      std::lock_guard<decltype(worker.mutex)> lock(worker.mutex);
      worker.should_exit = true;
      worker.promise.set_value();
    }

    for (auto& thread : threads_) {
      thread.join();
    }
  }

  std::future<void> submitToWorker(Task&& task, const size_t worker_idx) {
    CHECK_LT(worker_idx, workers_.size());
    auto& worker = workers_[worker_idx];
    std::unique_lock lock(worker.mutex);
    auto future = task.get_future();
    worker.tasks.push_back(std::move(task));
    try {
      worker.promise.set_value();
    } catch (const std::future_error& e) {
      if (e.code() != std::future_errc::promise_already_satisfied) {
        throw e;
      }
    }
    lock.unlock();
    return future;
  }

  std::future<void> submit(Task&& task) {
    const size_t worker_idx = dist_(prng_);
    return submitToWorker(std::move(task), worker_idx);
  }

  std::vector<std::future<void>> submitBatch(std::vector<Task>&& tasks) {
    auto clock_begin = timer_start();
    std::vector<std::future<void>> futures;
    const size_t workers_size = workers_.size() == 1 ? 1 : workers_.size() - 1;
    for (size_t i = 0; i < tasks.size(); i++) {
      const size_t worker_idx = i & workers_size;
      futures.emplace_back(submitToWorker(std::move(tasks[i]), worker_idx));
    }
    VLOG(1) << "Submitted all kernels in " << timer_stop(clock_begin);
    return futures;
  }

  size_t numWorkers() const { return threads_.size(); }

 private:
  void worker(const size_t thread_idx) {
    CHECK_LT(thread_idx, workers_.size());
    auto& worker = workers_[thread_idx];

    while (true) {
      worker.future.wait();

      std::unique_lock ready_lock(worker.mutex);
      if (worker.should_exit) {
        return;
      }

      CHECK(!worker.tasks.empty());

      auto clock_begin = timer_start();

      auto tasks = std::move(worker.tasks);
      worker.tasks.clear();
      // reset promise
      worker.promise = std::promise<void>();
      worker.future = worker.promise.get_future();
      ready_lock.unlock();

      for (auto& task : tasks) {
        task();
      }

      VLOG(1) << "Thread " << thread_idx << " finished task in "
              << timer_stop(clock_begin) << ".";
    }
  }

  std::vector<std::thread> threads_;

  struct WorkerInterface {
    WorkerInterface() { future = promise.get_future(); }

    WorkerInterface(const WorkerInterface&) = delete;

    std::mutex mutex;
    std::promise<void> promise;
    std::future<void> future;
    std::deque<Task> tasks;
    bool should_exit{false};
  };
  std::vector<WorkerInterface> workers_;

  std::mt19937 prng_{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist_;
};
