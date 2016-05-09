#include "../StringDictionary/StringDictionary.h"

#include <limits>

#include <glog/logging.h>
#include <gtest/gtest.h>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

TEST(StringDictionary, AddAndGet) {
  StringDictionary string_dict(BASE_PATH, false);
  auto id1 = string_dict.getOrAdd("foo bar");
  auto id2 = string_dict.getOrAdd("foo bar");
  ASSERT_EQ(id1, id2);
  ASSERT_EQ(0, id1);
  auto id3 = string_dict.getOrAdd("baz");
  ASSERT_EQ(1, id3);
  auto id4 = string_dict.get("foo bar");
  ASSERT_EQ(id1, id4);
  ASSERT_EQ("foo bar", string_dict.getString(id4));
}

TEST(StringDictionary, Recover) {
  StringDictionary string_dict(BASE_PATH);
  auto id1 = string_dict.getOrAdd("baz");
  ASSERT_EQ(1, id1);
  auto id2 = string_dict.getOrAdd("baz");
  ASSERT_EQ(1, id2);
  auto id3 = string_dict.getOrAdd("foo bar");
  ASSERT_EQ(0, id3);
  auto id4 = string_dict.getOrAdd("fizzbuzz");
  ASSERT_EQ(2, id4);
  ASSERT_EQ("baz", string_dict.getString(id2));
  ASSERT_EQ("foo bar", string_dict.getString(id3));
  ASSERT_EQ("fizzbuzz", string_dict.getString(id4));
}

TEST(StringDictionary, HandleEmpty) {
  StringDictionary string_dict(BASE_PATH, false);
  auto id1 = string_dict.getOrAdd("");
  auto id2 = string_dict.getOrAdd("");
  ASSERT_EQ(id1, id2);
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), id1);
}

const int g_op_count{250000};

TEST(StringDictionary, ManyAddsAndGets) {
  StringDictionary string_dict(BASE_PATH, false);
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(i, string_dict.getOrAdd(std::to_string(i)));
  }
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(i, string_dict.getOrAdd(std::to_string(i)));
  }
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(std::to_string(i), string_dict.getString(i));
  }
}

TEST(StringDictionary, RecoverMany) {
  StringDictionary string_dict(BASE_PATH, true);
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(i, string_dict.getOrAdd(std::to_string(i)));
  }
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(std::to_string(i), string_dict.getString(i));
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  auto err = RUN_ALL_TESTS();
  return err;
}
