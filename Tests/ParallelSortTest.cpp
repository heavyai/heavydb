#include <algorithm>
#include <random>
#include "Shared/parallel_sort.h"
#include "TestHelpers.h"

TEST(ParallelSortTest, TestInsertionSort) {
  std::random_device rnd_device;
  std::mt19937 rnd{rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int> dist{-3000, 3000};

  auto gen = [&dist, &rnd]() { return dist(rnd); };

  for (int i = 0; i < 10; i++) {
    std::vector<int> keys(100000);
    std::vector<size_t> values(100000);
    std::generate(keys.begin(), keys.end(), gen);
    std::iota(values.begin(), values.end(), 0);

    auto expected = keys;
    std::sort(expected.begin(), expected.end());

    auto original_keys = keys;

    insertion_sort_by_key(keys.begin(), values.begin(), 100000, std::less<int>());

    for (int i = 0; i < 100000; i++) {
      EXPECT_EQ(expected[i], keys[i]);
      EXPECT_EQ(expected[i], original_keys[values[i]]);
    }
  }
}

TEST(ParallelSortTest, TestParallelSort) {
  std::random_device rnd_device;
  std::mt19937 rnd{rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int> dist{-3000, 3000};

  auto gen = [&dist, &rnd]() { return dist(rnd); };

  for (int i = 0; i < 10; i++) {
    std::vector<int> keys(100000);
    std::vector<size_t> values(100000);
    std::generate(keys.begin(), keys.end(), gen);
    std::iota(values.begin(), values.end(), 0);

    auto expected = keys;
    std::sort(expected.begin(), expected.end());

    auto original_keys = keys;

    parallel_sort_by_key(keys.begin(), values.begin(), 100000, std::less<int>());

    for (int i = 0; i < 100000; i++) {
      EXPECT_EQ(expected[i], keys[i]);
      EXPECT_EQ(expected[i], original_keys[values[i]]);
    }
  }
}

int main(int argc, char* argv[]) {
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
