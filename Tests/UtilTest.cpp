#include "../Utils/StringLike.h"
#include "gtest/gtest.h"

TEST(Utils, StringLike) {
  ASSERT_TRUE(string_like("abc", 3, "abc", 3, '\\'));
  ASSERT_FALSE(string_like("abc", 3, "ABC", 3, '\\'));
  ASSERT_TRUE(string_ilike("Xyzabc", 6, "xyz%", 4, '\\'));
  ASSERT_TRUE(string_like("abcxyzefg", 9, "%xyz%", 5, '\\'));
  ASSERT_TRUE(string_like("abcxyzefgXYZhij", 15, "%xyz%XYZ%", 9, '\\'));
  ASSERT_TRUE(string_like("abcxOzefgXpZhij", 15, "%x_z%X_Z%", 9, '\\'));
  ASSERT_TRUE(string_like("abc100%efg", 10, "%100!%___", 9, '!'));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
