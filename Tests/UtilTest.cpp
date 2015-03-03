#include "../Utils/StringLike.h"
#include "gtest/gtest.h"

TEST(Utils, StringLike) {
	ASSERT_TRUE(string_like("abc", 3, "abc", 3, '\\', false));
	ASSERT_FALSE(string_like("abc", 3, "ABC", 3, '\\', false));
	ASSERT_TRUE(string_like("abc", 3, "AbC", 3, '\\', true));
	ASSERT_TRUE(string_like("Xyzabc", 6, "xyz%", 4, '\\', true));
  ASSERT_TRUE(string_like("abcxyzefg", 9, "%xyz%", 5, '\\', false));
  ASSERT_TRUE(string_like("abcxyzefgXYZhij", 15, "%xyz%XYZ%", 9, '\\', false));
  ASSERT_TRUE(string_like("abcxOzefgXpZhij", 15, "%x_z%X_Z%", 9, '\\', false));
  ASSERT_TRUE(string_like("abc100%efg", 10, "%100!%___", 9, '!', false));
}

int
main(int argc, char* argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
