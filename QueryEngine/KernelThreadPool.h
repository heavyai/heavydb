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

#include <atomic>
#include <deque>
#include <future>
#include <memory>
#include <random>

#include "Logger/Logger.h"
#include "Shared/measure.h"
#include "Shared/scope.h"

/**
 * A thread pool which supports thread re-use across tasks.  Tasks are submitted to the
 * queue. Notification is submitted to a worker thread for each submitted task to the
 * pool. An open thread will then pick up the task.
 *
 */
class KernelThreadPool {
 public:
  using Task = std::packaged_task<void(size_t)>;  // void task(const size_t thread_idx)

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
    while (worker.lock.test_and_set(std::memory_order_acquire))
      ;  // spin
    ScopeGuard unlock = [&] { worker.lock.clear(); };
    return submitToWorkerUnlocked(std::move(task), worker);
  }

  std::future<void> submit(Task&& task) {
    const size_t worker_idx = dist_(prng_);
    return submitToWorker(std::move(task), worker_idx);
  }

  std::vector<std::future<void>> submitBatch(std::vector<Task>&& tasks) {
    auto clock_begin = timer_start();
    std::vector<std::future<void>> futures;

    for (size_t worker_idx = 0; worker_idx < workers_.size(); worker_idx++) {
      // take every worker-th task
      for (size_t j = worker_idx; j < tasks.size(); j += workers_.size()) {
        futures.emplace_back(submitToWorker(std::move(tasks[j]), worker_idx));
      }
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

      if (worker.should_exit) {
        return;
      }

      auto clock_begin = timer_start();

      while (worker.lock.test_and_set(std::memory_order_acquire))
        ;  // spin

      // reset promise
      worker.promise = std::promise<void>();
      worker.future = worker.promise.get_future();

      if (!worker.pending_tasks.empty()) {
        auto tasks = std::move(worker.pending_tasks);
        worker.pending_tasks.clear();
        worker.lock.clear(std::memory_order_release);

        for (auto& task : tasks) {
          task(thread_idx);
        }

        VLOG(1) << "Thread " << thread_idx << " finished task in "
                << timer_stop(clock_begin) << ".";
      }
    }

    UNREACHABLE();
  }

  struct WorkerInterface {
    WorkerInterface() { future = promise.get_future(); }

    WorkerInterface(const WorkerInterface&) = delete;

    std::promise<void> promise;
    std::future<void> future;
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    std::deque<Task> pending_tasks;
    std::atomic<bool> should_exit{false};
  };

  std::future<void> submitToWorkerUnlocked(Task&& task, WorkerInterface& worker) {
    auto future = task.get_future();
    worker.pending_tasks.push_back(std::move(task));
    try {
      worker.promise.set_value();
    } catch (const std::future_error& e) {
      if (e.code() != std::future_errc::promise_already_satisfied) {
        throw e;
      }
    }
    return future;
  }

  std::vector<std::thread> threads_;
  std::vector<WorkerInterface> workers_;

  std::mt19937 prng_{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist_;
};
