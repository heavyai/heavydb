#include "../Shared/ConstExprLib.h"
#include "../Shared/ExperimentalTypeUtilities.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <exception>

using namespace Experimental;

constexpr ConstExprPair<int, int> yield_test_constexpr_pair() {
  ConstExprPair<int, int> default_construct1, default_construct2(1, 2);
  if (!(default_construct1.first == 0 && default_construct1.second == 0)) {
    throw std::runtime_error("Faulty construction");
  }
  default_construct1.swap(default_construct2);
  if (!(default_construct1.first == 1 && default_construct1.second == 2)) {
    throw std::runtime_error("Faulty swap");
  }
  ConstExprPair<int, int> component_construct1(1, 1), component_construct2(2, 2);
  component_construct1 = component_construct2;
  if (!(component_construct1.first == 2 && component_construct1.second == 2)) {
    throw std::runtime_error("Faulty construction");
  }
  ConstExprPair<int, int> component_construct3(component_construct2);
  if (!(component_construct3.first == 2 && component_construct3.second == 2)) {
    throw std::runtime_error("Faulty component construction");
  }

  return component_construct3;
}

TEST(ExperimentalTest, ConstExprLib_Pair) {
  // Verify run-time use
  ConstExprPair<int, int> default_construct1, default_construct2(1, 2);
  ASSERT_EQ(default_construct1.first, 0);
  ASSERT_EQ(default_construct1.second, 0);

  default_construct1.swap(default_construct2);
  ASSERT_EQ(default_construct1.first, 1);
  ASSERT_EQ(default_construct1.second, 2);

  ConstExprPair<int, int> component_construct1(1, 1), component_construct2(2, 2);
  component_construct1 = component_construct2;
  ASSERT_EQ(component_construct1.first, 2);
  ASSERT_EQ(component_construct1.second, 2);

  ConstExprPair<int, int> component_construct3(component_construct2);
  ASSERT_EQ(component_construct3.first, 2);
  ASSERT_EQ(component_construct3.second, 2);

  // Verify compile time use
  constexpr auto compile_time_verify = yield_test_constexpr_pair();
  ASSERT_EQ(compile_time_verify.first, 2);
  ASSERT_EQ(compile_time_verify.second, 2);
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
