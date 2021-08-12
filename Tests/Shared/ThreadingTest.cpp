#include "Tests/TestHelpers.h"
#include <gtest/gtest.h>

#include "Shared/threading.h"
#include <atomic>

std::atomic<int> g_counter {0};

#define MODED(a) a##std
#include "ThreadingTest.h"

#if HAVE_TBB
#undef MODED
#define MODED(a) a##tbb
#include "ThreadingTest.h"
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
