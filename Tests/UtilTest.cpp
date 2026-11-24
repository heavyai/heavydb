/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/Intervals.h"
#include "Shared/JsonUtils.h"
#include "Shared/StringTransform.h"
#include "TestHelpers.h"
#include "Utils/Regexp.h"
#include "Utils/StringLike.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <future>

namespace {
template <typename T>
void check_json_round_trip(const T& expected) {
  rapidjson::Document document;
  json_utils::set_value(document, expected, document.GetAllocator());
  const auto serialized = json_utils::write_to_string(document);
  const auto parsed = json_utils::read_from_string(serialized);
  ASSERT_FALSE(parsed.HasParseError());
  T actual{};
  json_utils::get_value(parsed, actual);
  EXPECT_EQ(expected, actual);
}
}  // namespace

TEST(JsonUtils, ScalarRoundTrips) {
  check_json_round_trip(std::numeric_limits<size_t>::max());
  check_json_round_trip(std::numeric_limits<int>::min());
  check_json_round_trip(std::numeric_limits<int>::max());
  check_json_round_trip(std::numeric_limits<int64_t>::min());
  check_json_round_trip(std::numeric_limits<int64_t>::max());
  check_json_round_trip(std::string("quoted \"column\""));
  check_json_round_trip(std::string{});
  check_json_round_trip(true);
  check_json_round_trip(false);
  check_json_round_trip(kTEXT);
  check_json_round_trip(kENCODING_DICT);
}

TEST(JsonUtils, ScalarTypeErrorsThrow) {
  const auto check_invalid = [](auto original, const auto& document) {
    auto result = original;
    EXPECT_THROW(json_utils::get_value(document, result), json_utils::json_schema_error);
    EXPECT_EQ(original, result);
  };
  for (const auto input : {"null", "{}", "[]"}) {
    SCOPED_TRACE(input);
    const auto document = json_utils::read_from_string(input);
    check_invalid(size_t(1), document);
    check_invalid(1, document);
    check_invalid(int64_t(1), document);
    check_invalid(std::string("unchanged"), document);
    check_invalid(true, document);
    check_invalid(kTEXT, document);
    check_invalid(kENCODING_DICT, document);
  }
  check_invalid(size_t(1), json_utils::read_from_string("-1"));
  check_invalid(1, json_utils::read_from_string("2147483648"));
  check_invalid(int64_t(1), json_utils::read_from_string("18446744073709551615"));
  check_invalid(1, json_utils::read_from_string("1.5"));
  check_invalid(std::string("unchanged"), json_utils::read_from_string("1"));
  check_invalid(true, json_utils::read_from_string("1"));
}

TEST(JsonUtils, MemberErrorsThrow) {
  for (const auto input : {"null", "[]", "true", "42", "\"text\"", "{}"}) {
    SCOPED_TRACE(input);
    const auto document = json_utils::read_from_string(input);
    int result = 7;
    EXPECT_THROW(json_utils::get_member(document, "missing"),
                 json_utils::json_schema_error);
    EXPECT_THROW(json_utils::get_value_from_object(document, result, "missing"),
                 json_utils::json_schema_error);
    EXPECT_EQ(7, result);
  }
  const auto document = json_utils::read_from_string(R"({"id":42,"name":null})");
  EXPECT_EQ(42, json_utils::get_int_member(document, "id"));
  EXPECT_THROW(json_utils::get_int_member(document, "name"),
               json_utils::json_schema_error);
  try {
    json_utils::get_member(document, "missing");
    FAIL() << "Missing member was accepted";
  } catch (const json_utils::json_schema_error& e) {
    EXPECT_NE(std::string::npos, std::string(e.what()).find("missing"));
  }
}

TEST(JsonUtils, ContainerRoundTrips) {
  check_json_round_trip(std::vector<std::string>{"id", "parent_id"});
  check_json_round_trip(std::list<std::string>{"id", "parent_id"});
  check_json_round_trip(std::vector<std::pair<int, std::string>>{{1, "id"}, {2, "name"}});
  check_json_round_trip(std::list<std::pair<int, std::string>>{{1, "id"}, {2, "name"}});
  check_json_round_trip(std::map<int, std::string>{{1, "id"}, {2, "name"}});
  check_json_round_trip(std::vector<std::string>{});
  check_json_round_trip(std::list<std::string>{});
  check_json_round_trip(std::vector<std::pair<int, std::string>>{});
  check_json_round_trip(std::list<std::pair<int, std::string>>{});
  check_json_round_trip(std::map<int, std::string>{});
}

TEST(JsonUtils, ContainerErrorsDoNotPublishPartialResults) {
  const auto check_invalid = [](const auto& document, auto output) {
    EXPECT_THROW(json_utils::get_value(document, output), json_utils::json_schema_error);
    EXPECT_TRUE(output.empty());
  };
  for (const auto input : {"null", "{}", "true", "42", "\"id\""}) {
    SCOPED_TRACE(input);
    const auto document = json_utils::read_from_string(input);
    check_invalid(document, std::vector<std::string>{});
    check_invalid(document, std::list<std::string>{});
    check_invalid(document, std::vector<std::pair<int, std::string>>{});
    check_invalid(document, std::list<std::pair<int, std::string>>{});
    check_invalid(document, std::map<int, std::string>{});
  }
  for (const auto input : {R"(["id",42])", R"(["id",null])"}) {
    SCOPED_TRACE(input);
    const auto document = json_utils::read_from_string(input);
    check_invalid(document, std::vector<std::string>{});
    check_invalid(document, std::list<std::string>{});
  }
  for (const auto input : {R"([{"key":1,"value":"id"},null])",
                           R"([{"key":1,"value":"id"},{"value":"name"}])",
                           R"([{"key":1,"value":"id"},{"key":2}])",
                           R"([{"key":1,"value":"id"},{"key":"bad","value":"name"}])",
                           R"([{"key":1,"value":"id"},{"key":2,"value":42}])"}) {
    SCOPED_TRACE(input);
    const auto document = json_utils::read_from_string(input);
    check_invalid(document, std::vector<std::pair<int, std::string>>{});
    check_invalid(document, std::list<std::pair<int, std::string>>{});
    check_invalid(document, std::map<int, std::string>{});
  }
}

TEST(JsonUtils, IntegerArrayUsesSharedValidation) {
  const auto document = json_utils::read_from_string("[1,-2,3]");
  EXPECT_EQ((std::vector<int32_t>{1, -2, 3}), json_utils::get_int_vector(document));
  for (const auto input : {"null", "{}", "[1,null]", "[1,2147483648]"}) {
    SCOPED_TRACE(input);
    const auto document = json_utils::read_from_string(input);
    EXPECT_THROW(json_utils::get_int_vector(document), json_utils::json_schema_error);
  }
}

TEST(JsonUtils, MetadataRoundTrips) {
  check_json_round_trip(shared::StringDictKey(1, 2));
  check_json_round_trip(SQLTypeInfo(kINT, false));
  check_json_round_trip(SQLTypeInfo(kDECIMAL, 12, 3, false));
}

TEST(JsonUtils, MetadataErrorsThrow) {
  for (const auto input : {"null", "[]", "{}", R"({"type":"bad"})"}) {
    SCOPED_TRACE(input);
    const auto document = json_utils::read_from_string(input);
    SQLTypeInfo type;
    EXPECT_THROW(json_utils::get_value(document, type), json_utils::json_schema_error);
  }
  for (const auto input : {"null", "[]", "{}", R"({"db_id":1,"dict_id":null})"}) {
    SCOPED_TRACE(input);
    const auto document = json_utils::read_from_string(input);
    shared::StringDictKey key;
    EXPECT_THROW(json_utils::get_value(document, key), json_utils::json_schema_error);
  }
}

// for (auto const interval : makeIntervals(0, M, n_workers)) {...}
// iterates over interval={begin,end} pairs which satisfy:
// Requirement_1) Mutually exclusive and complete coverage over [0,M).
// Requirement_2) Difference between max(end-begin) and min(end-begin) is either 0 or 1.
TEST(Shared, Intervals) {
  constexpr int M = 50;
  std::array<std::atomic<int>, M> array;
  for (int n_workers = 1; n_workers <= 2 * M; ++n_workers) {
    std::for_each(array.begin(), array.end(), [](auto& v) { v = 0; });
    std::vector<std::future<void>> threads;
    int max_interval_size = std::numeric_limits<int>::min();
    int min_interval_size = std::numeric_limits<int>::max();
    for (auto const interval : makeIntervals(0, M, n_workers)) {
      EXPECT_LT(interval.begin, interval.end);
      max_interval_size = std::max(max_interval_size, interval.end - interval.begin);
      min_interval_size = std::min(min_interval_size, interval.end - interval.begin);
      threads.push_back(std::async(
          std::launch::async,
          [&array](Interval<int> interval) {
            for (int i = interval.begin; i < interval.end; ++i) {
              ++array.at(i);
            }
          },
          interval));
    }
    for (auto& thread : threads) {
      thread.wait();
    }
    // Test Requirement_1
    for (auto const& v : array) {
      EXPECT_EQ(v, 1);
    }
    // Test Requirement_2
    EXPECT_LE(0, max_interval_size - min_interval_size);
    EXPECT_LE(max_interval_size - min_interval_size, 1);
    // If M < n_workers, then there is no need to spawn unneeded workers.
    EXPECT_LE(threads.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
    EXPECT_LE(static_cast<int>(threads.size()), std::min(M, n_workers));
  }
}

TEST(Shared, IntervalsBounds) {
  using Integer = int8_t;
  using Unsigned = std::make_unsigned<Integer>::type;
  constexpr int M = 255;  // Number of elements in [-128,127)
  std::array<std::atomic<int>, M> array;
  // Test with 0-value, and values that go outside the range of Integer.
  for (unsigned n_workers = 0; n_workers <= 260; ++n_workers) {
    std::for_each(array.begin(), array.end(), [](auto& v) { v = 0; });
    std::vector<std::future<void>> threads;
    Integer const begin = std::numeric_limits<Integer>::min();
    Integer const end = std::numeric_limits<Integer>::max();
    int const offset = -static_cast<int>(begin);
    Unsigned max_interval_size = std::numeric_limits<Unsigned>::min();
    Unsigned min_interval_size = std::numeric_limits<Unsigned>::max();
    for (auto const interval : makeIntervals(begin, end, n_workers)) {
      EXPECT_LT(interval.begin, interval.end);
      max_interval_size = std::max(max_interval_size,
                                   static_cast<Unsigned>(interval.end - interval.begin));
      min_interval_size = std::min(min_interval_size,
                                   static_cast<Unsigned>(interval.end - interval.begin));
      threads.push_back(std::async(
          std::launch::async,
          [&array](Interval<Integer> interval) {
            for (Integer i = interval.begin; i < interval.end; ++i) {
              ++array.at(offset + i);
            }
          },
          interval));
    }
    for (auto& thread : threads) {
      thread.wait();
    }
    // Test Requirement_1
    for (auto const& v : array) {
      EXPECT_EQ(v, int(0 != n_workers));
    }
    // Test Requirement_2
    if (n_workers) {
      EXPECT_LE(0, max_interval_size - min_interval_size);
      EXPECT_LE(max_interval_size - min_interval_size, 1);
    } else {  // unchanged
      EXPECT_EQ(max_interval_size, std::numeric_limits<Unsigned>::min());
      EXPECT_EQ(min_interval_size, std::numeric_limits<Unsigned>::max());
    }
    // If M < n_workers, then there is no need to spawn unneeded workers.
    EXPECT_LE(threads.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
    EXPECT_LE(static_cast<int>(threads.size()), std::min(M, static_cast<int>(n_workers)));
  }
}

TEST(Shared, IntervalsInvalidBounds) {
  using Integer = int;
  using Unsigned = std::make_unsigned<Integer>::type;
  bool loop_body_executed = false;
  // Expect loop body is never executed since begin == end.
  for (auto const interval : makeIntervals(10, 10, 1)) {
    (void)interval;
    loop_body_executed = true;
  }
  EXPECT_FALSE(loop_body_executed);
  // Expect loop body is never executed since begin > end.
  for (auto const interval : makeIntervals(10, 9, 1)) {
    (void)interval;
    loop_body_executed = true;
  }
  EXPECT_FALSE(loop_body_executed);
  // Expect loop body is never executed since n_workers == 0.
  for (auto const interval : makeIntervals(9, 10, 0)) {
    (void)interval;
    loop_body_executed = true;
  }
  EXPECT_FALSE(loop_body_executed);
  // Sanity check that loop body is entered with valid values.
  for (auto const interval : makeIntervals(9, 10, 1)) {
    (void)interval;
    loop_body_executed = true;
  }
  EXPECT_TRUE(loop_body_executed);
}

TEST(Shared, StringViewStrip) {
  std::vector<std::pair<std::string_view, std::string_view>> svs = {
      {"", ""},
      {"   ", ""},
      {" ", ""},
      {"  abcd", "abcd"},
      {"abcd  ", "abcd"},
      {"  abcd   ", "abcd"},
      {"  a   b c  d   ", "a   b c  d"},
  };
  for (auto& [actual, expected] : svs) {
    ASSERT_EQ(sv_strip(actual), expected);
  }
}

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
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
