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

#include "TestHelpers.h"

#include <benchmark/benchmark.h>

#include "../QueryEngine/KernelThreadPool.h"

std::once_flag setup_flag;
void global_setup() {
  TestHelpers::init_logger_stderr_only();
}

class ThreadPoolFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);
  }

  void TearDown(const ::benchmark::State& state) override {
    // noop
  }

  static std::unique_ptr<KernelThreadPool> thread_pool;
};

BENCHMARK_DEFINE_F(ThreadPoolFixture, TasksEqualThreadsBatch)(benchmark::State& state) {
  KernelThreadPool thread_pool(std::thread::hardware_concurrency());
  std::vector<std::future<void>> execution_kernel_futures;

  for (auto _ : state) {
    const auto thread_pool_size = thread_pool.numWorkers();
    std::vector<KernelThreadPool::Task> tasks;
    tasks.reserve(thread_pool_size);
    for (size_t i = 0; i < thread_pool_size; i++) {
      tasks.emplace_back([](const size_t thread_idx) {
        int x = 0;
        for (size_t i = 0; i < 1000000; i++) {
          benchmark::DoNotOptimize(x = x + 1);
        }
      });
    }
    execution_kernel_futures = thread_pool.submitBatch(std::move(tasks));
    // wait first to ensure all threads exit before reading exceptions
    for (auto& future : execution_kernel_futures) {
      future.wait();
    }

    // get() reads exceptions and may terminate early. the clear queue scope guard will
    // ensure all tasks get cleaned up upon termination.
    for (auto& future : execution_kernel_futures) {
      future.get();
    }
  }
}

BENCHMARK_REGISTER_F(ThreadPoolFixture, TasksEqualThreadsBatch)
    ->Range(100, 1000000)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(ThreadPoolFixture, Tasks10xThreadsBatch)(benchmark::State& state) {
  KernelThreadPool thread_pool(std::thread::hardware_concurrency());
  std::vector<std::future<void>> execution_kernel_futures;

  for (auto _ : state) {
    const auto thread_pool_size = thread_pool.numWorkers();
    std::vector<KernelThreadPool::Task> tasks;
    tasks.reserve(10 * thread_pool_size);
    for (size_t i = 0; i < 10 * thread_pool_size; i++) {
      tasks.emplace_back([](const size_t thread_idx) {
        int x = 0;
        for (size_t i = 0; i < 1000000; i++) {
          benchmark::DoNotOptimize(x = x + 1);
        }
      });
    }
    execution_kernel_futures = thread_pool.submitBatch(std::move(tasks));
    // wait first to ensure all threads exit before reading exceptions
    for (auto& future : execution_kernel_futures) {
      future.wait();
    }

    // get() reads exceptions and may terminate early. the clear queue scope guard will
    // ensure all tasks get cleaned up upon termination.
    for (auto& future : execution_kernel_futures) {
      future.get();
    }
  }
}

BENCHMARK_REGISTER_F(ThreadPoolFixture, Tasks10xThreadsBatch)
    ->Range(100, 1000000)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
