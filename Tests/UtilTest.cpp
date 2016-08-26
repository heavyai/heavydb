#include "../Utils/StringLike.h"
#include "../Utils/Regexp.h"
#include "gtest/gtest.h"

TEST(Utils, StringLike) {
  ASSERT_TRUE(string_like("abc", 3, "abc", 3, '\\'));
  ASSERT_FALSE(string_like("abc", 3, "ABC", 3, '\\'));
  ASSERT_TRUE(string_ilike("Xyzabc", 6, "xyz%", 4, '\\'));
  ASSERT_TRUE(string_like("abcxyzefg", 9, "%xyz%", 5, '\\'));
  ASSERT_TRUE(string_like("abcxyzefgXYZhij", 15, "%xyz%XYZ%", 9, '\\'));
  ASSERT_TRUE(string_like("abcxOzefgXpZhij", 15, "%x_z%X_Z%", 9, '\\'));
  ASSERT_TRUE(string_like("abc100%efg", 10, "%100!%___", 9, '!'));
  ASSERT_TRUE(string_like("[ hello", 7, "%\\[%", 4, '\\'));
  ASSERT_TRUE(string_like("hello [", 7, "%\\[%", 4, '\\'));
}

TEST(Utils, Regexp) {
  ASSERT_TRUE(regexp_like("abc", 3, "abc", 3, '\\'));
  ASSERT_FALSE(regexp_like("abc", 3, "ABC", 3, '\\'));
  ASSERT_TRUE(regexp_like("Xyzabc", 6, "[xX]yz.*", 8, '\\'));
  ASSERT_TRUE(regexp_like("abcxyzefg", 9, ".*xyz.*", 7, '\\'));
  ASSERT_TRUE(regexp_like("abcxyzefgXYZhij", 15, ".*xyz.*XYZ.*", 12, '\\'));
  ASSERT_TRUE(regexp_like("abcxOzefgXpZhij", 15, ".+x.z.*X.Z.*", 12, '\\'));
  // Custom escape characters are not yet supported.
  ASSERT_FALSE(regexp_like("abc100%efg", 10, ".+100!%...", 10, '!'));
  ASSERT_TRUE(regexp_like("[ hello", 7, ".*\\[.*", 6, '\\'));
  ASSERT_TRUE(regexp_like("hello [", 7, ".*\\[.*", 6, '\\'));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
