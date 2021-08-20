#include <gtest/gtest.h>
#include "Tests/TestHelpers.h"

#include <atomic>
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

TEST(THREADING, TaskGroup) {
  using namespace threading;
  g_counter = 0;
  task_group tg;
  tg.run([&] { g_counter++; });
  tg.wait();
  ASSERT_EQ(g_counter, 1);
}

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
