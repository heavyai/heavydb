/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include "Tests/TestHelpers.h"

#include <atomic>
#include <numeric>
#include <vector>

#include "Shared/threading.h"

std::atomic<int> g_counter{0};

#if DISABLE_CONCURRENCY
#define THREADING THREADING_SERIAL
#elif ENABLE_TBB
#define THREADING THREADING_TBB
#else
#define THREADING THREADING_STD
#endif

TEST(THREADING, ParallelFor) {
  using namespace threading;
  g_counter = 0;
  parallel_for(1, 100, [&](int i) { g_counter++; });
  ASSERT_EQ(g_counter, 99);
  g_counter = 0;
  parallel_for(blocked_range<size_t>(1, 10000), [&](auto r) { g_counter += r.size(); });
  ASSERT_EQ(g_counter, 9999);
  g_counter = 0;
}

TEST(THREADING, ParallelForPublishesBoundState) {
  using namespace threading;
  constexpr size_t value_count = 64;
  std::vector<size_t> values(value_count);

  parallel_for(
      size_t(0), value_count, [&](const size_t index) { values[index] = index + 1; });

  EXPECT_EQ(std::accumulate(values.begin(), values.end(), size_t(0)),
            value_count * (value_count + 1) / 2);
}

TEST(THREADING, ParallelReduce) {
  using namespace threading;
  g_counter = 0;
  int res = parallel_reduce(
      blocked_range<size_t>(1, 10000),
      int(0),
      [&](auto r, int v) {
        g_counter += r.size();
        return int(v + r.size());
      },
      std::plus<int>());
  ASSERT_EQ(g_counter, res);
  g_counter = 0;
}

TEST(THREADING, Async) {
  using namespace threading;
  g_counter = 0;
  auto a1 = async(
      [&](int i) {
        g_counter++;
        ASSERT_EQ(i, 1);
      },
      1);
  a1.wait();
  ASSERT_EQ(g_counter, 1);
}

TEST(THREADING, AsyncPublishesBoundState) {
  using namespace threading;
  constexpr size_t task_count = 64;
  std::vector<size_t> values(task_count);
  std::vector<future<void>> tasks;
  tasks.reserve(task_count);

  auto warmup = async([] {});
  warmup.get();
  for (size_t i = 0; i < task_count; ++i) {
    tasks.emplace_back(async([&values, i] { values[i] = i + 1; }));
  }
  for (auto& task : tasks) {
    task.get();
  }

  EXPECT_EQ(std::accumulate(values.begin(), values.end(), size_t(0)),
            task_count * (task_count + 1) / 2);
}

TEST(THREADING, TaskGroup) {
  using namespace threading;
  g_counter = 0;
  task_group tg;
  tg.run([&] { g_counter++; });
  tg.wait();
  ASSERT_EQ(g_counter, 1);
}

TEST(THREADING, TaskGroupPublishesBoundState) {
  using namespace threading;
  constexpr size_t task_count = 64;
  std::vector<size_t> values(task_count);
  task_group tasks;

  for (size_t i = 0; i < task_count; ++i) {
    tasks.run([&values, i] { values[i] = i + 1; });
  }
  tasks.wait();

  EXPECT_EQ(std::accumulate(values.begin(), values.end(), size_t(0)),
            task_count * (task_count + 1) / 2);
}

#if ENABLE_TBB
TEST(THREADING, AsyncPublishesNestedTbbWork) {
  constexpr size_t value_count = 4096;
  std::vector<size_t> values(value_count);

  auto task = threading::async([&values] {
    tbb::parallel_for(size_t(0), values.size(), [&values](const size_t index) {
      values[index] = index + 1;
    });
  });
  task.get();

  EXPECT_EQ(std::accumulate(values.begin(), values.end(), size_t(0)),
            value_count * (value_count + 1) / 2);
}

TEST(THREADING, TaskGroupSupportsNestedSubmission) {
  using namespace threading;
  constexpr size_t outer_task_count = 8;
  constexpr size_t inner_task_count = 8;
  constexpr size_t value_count = outer_task_count * inner_task_count;
  std::vector<size_t> values(value_count);
  task_group tasks;

  for (size_t outer = 0; outer < outer_task_count; ++outer) {
    tasks.run([&tasks, &values, outer] {
      for (size_t inner = 0; inner < inner_task_count; ++inner) {
        const auto index = outer * inner_task_count + inner;
        tasks.run([&values, index] { values[index] = index + 1; });
      }
    });
  }
  tasks.wait();

  EXPECT_EQ(std::accumulate(values.begin(), values.end(), size_t(0)),
            value_count * (value_count + 1) / 2);
}
#endif

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
