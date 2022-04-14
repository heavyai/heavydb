/*
 * Copyright 2019 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file StringFunctionsTest.cpp
 * @brief Test suite for string functions
 */

#include "Shared/ArrowSQLRunner/ArrowSQLRunner.h"
#include "TestHelpers.h"

#include <QueryEngine/ResultSet.h>

#include <gtest/gtest.h>
#include <boost/format.hpp>
#include <boost/locale/generator.hpp>

extern bool g_enable_experimental_string_functions;

using namespace TestHelpers;
using namespace TestHelpers::ArrowSQLRunner;

namespace {

class AssertValueEqualsVisitor : public boost::static_visitor<> {
 public:
  AssertValueEqualsVisitor(const size_t& row, const size_t& column)
      : row(row), column(column) {}

  template <typename T, typename U>
  void operator()(const T& expected, const U& actual) const {
    FAIL() << boost::format(
                  "Values are of different types. Expected result set value: %s is of "
                  "type: %s while actual result set value: %s is of type: %s. At row: "
                  "%d, column: %d") %
                  expected % typeid(expected).name() % actual % typeid(actual).name() %
                  row % column;
  }

  template <typename T>
  void operator()(const T& expected, const T& actual) const {
    EXPECT_EQ(expected, actual) << boost::format("At row: %d, column: %d") % row % column;
  }

 private:
  size_t row;
  size_t column;
};

template <>
void AssertValueEqualsVisitor::operator()<NullableString>(
    const NullableString& expected,
    const NullableString& actual) const {
  boost::apply_visitor(AssertValueEqualsVisitor(row, column), expected, actual);
}

void assert_value_equals(ScalarTargetValue& expected,
                         ScalarTargetValue& actual,
                         const size_t& row,
                         const size_t& column) {
  boost::apply_visitor(AssertValueEqualsVisitor(row, column), expected, actual);
}

void compare_result_set(
    const std::vector<std::vector<ScalarTargetValue>>& expected_result_set,
    const std::shared_ptr<ResultSet>& actual_result_set) {
  auto row_count = actual_result_set->rowCount(false);
  ASSERT_EQ(expected_result_set.size(), row_count)
      << "Returned result set does not have the expected number of rows";

  if (row_count == 0) {
    return;
  }

  auto expected_column_count = expected_result_set[0].size();
  auto column_count = actual_result_set->colCount();
  ASSERT_EQ(expected_column_count, column_count)
      << "Returned result set does not have the expected number of columns";
  ;

  for (size_t r = 0; r < row_count; ++r) {
    auto row = actual_result_set->getNextRow(true, true);
    for (size_t c = 0; c < column_count; c++) {
      auto column_value = boost::get<ScalarTargetValue>(row[c]);
      auto expected_column_value = expected_result_set[r][c];
      assert_value_equals(expected_column_value, column_value, r, c);
    }
  }
}
}  // namespace

// begin LOWER function tests

/**
 * @brief Class used for setting up and tearing down tables and records that are required
 * by the LOWER function test cases
 */
class LowerFunctionTest : public testing::Test {
 public:
  void SetUp() override {
    createTable("lower_function_test_people",
                {{"first_name", dictType()},
                 {"last_name", SQLTypeInfo(kTEXT)},
                 {"age", SQLTypeInfo(kINT)},
                 {"country_code", dictType()}});
    insertCsvValues(
        "lower_function_test_people",
        "JOHN,SMITH,25,us\nJohn,Banks,30,Us\nJOHN,Wilson,20,cA\nSue,Smith,25,CA");

    createTable(
        "lower_function_test_countries",
        {{"code", dictType()}, {"name", dictType()}, {"text", SQLTypeInfo(kTEXT)}});
    insertCsvValues("lower_function_test_countries",
                    "US,United States,Washington\nca,Canada,Ottawa\nGb,United "
                    "Kingdom,London\ndE,Germany,Berlin");
  }

  void TearDown() override {
    dropTable("lower_function_test_people");
    dropTable("lower_function_test_countries");
  }
};

TEST_F(LowerFunctionTest, LowercaseProjection) {
  auto result_set =
      run_multiple_agg("select lower(first_name) from lower_function_test_people;",
                       ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"john"}, {"john"}, {"john"}, {"sue"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, LowercaseFilter) {
  auto result_set = run_multiple_agg(
      "select first_name, last_name from lower_function_test_people "
      "where lower(country_code) = 'us';",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"JOHN", "SMITH"},
                                                                  {"John", "Banks"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, MultipleLowercaseFilters) {
  auto result_set = run_multiple_agg(
      "select first_name, last_name from lower_function_test_people "
      "where lower(country_code) = 'us' or lower(first_name) = 'sue';",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"JOHN", "SMITH"}, {"John", "Banks"}, {"Sue", "Smith"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, MixedFilters) {
  auto result_set = run_multiple_agg(
      "select first_name, last_name from lower_function_test_people "
      "where lower(country_code) = 'ca' and age > 20;",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Sue", "Smith"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, LowercaseGroupBy) {
  auto result_set = run_multiple_agg(
      "select lower(first_name), count(*) from lower_function_test_people "
      "group by lower(first_name) order by 2 desc;",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"john", int64_t(3)},
                                                                  {"sue", int64_t(1)}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, LowercaseJoin) {
  auto result_set = run_multiple_agg(
      "select first_name, name as country_name "
      "from lower_function_test_people "
      "join lower_function_test_countries on lower(country_code) = lower(code);",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"JOHN", "United States"},
      {"John", "United States"},
      {"JOHN", "Canada"},
      {"Sue", "Canada"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, SelectLowercaseLiteral) {
  auto result_set = run_multiple_agg(
      "select first_name, lower('SMiTH') from lower_function_test_people;",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"JOHN", "smith"}, {"John", "smith"}, {"JOHN", "smith"}, {"Sue", "smith"}};
  compare_result_set(expected_result_set, result_set);
}

// TODO: Re-enable after clear definition around handling non-ASCII characters
TEST_F(LowerFunctionTest, DISABLED_LowercaseNonAscii) {
  insertCsvValues("lower_function_test_people", "Ħ,Ħ,25,GB");
  auto result_set = run_multiple_agg(
      "select lower(first_name), last_name from lower_function_test_people where "
      "country_code = 'GB';",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"ħ", "Ħ"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, LowercaseNonEncodedTextColumn) {
  try {
    run_multiple_agg("select lower(last_name) from lower_function_test_people;",
                     ExecutorDeviceType::CPU);
    FAIL() << "An exception should have been thrown for this test case";
  } catch (const std::exception& e) {
    ASSERT_STREQ("LOWER expects a dictionary encoded text column or a literal.",
                 e.what());
  }
}

TEST_F(LowerFunctionTest, LowercaseNonTextColumn) {
  try {
    run_multiple_agg("select lower(age) from lower_function_test_people;",
                     ExecutorDeviceType::CPU);
    FAIL() << "An exception should have been thrown for this test case";
  } catch (const std::exception& e) {
    ASSERT_STREQ("LOWER expects a dictionary encoded text column or a literal.",
                 e.what());
  }
}

TEST_F(LowerFunctionTest, LowercaseGpuMode) {
#ifndef HAVE_CUDA
  LOG(ERROR)
      << "This test case only applies to uses case where CUDA is enabled. Skipping test.";
  return;
#else
  auto result_set = run_multiple_agg(
      "select first_name, last_name from lower_function_test_people "
      "where lower(country_code) = 'us';",
      ExecutorDeviceType::GPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"JOHN", "SMITH"},
                                                                  {"John", "Banks"}};
  compare_result_set(expected_result_set, result_set);
#endif
}

TEST_F(LowerFunctionTest, LowercaseNullColumn) {
  insertCsvValues("lower_function_test_people", ",Empty,25,US");
  auto result_set = run_multiple_agg(
      "select lower(first_name), last_name from lower_function_test_people where "
      "last_name = 'Empty';",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"", "Empty"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, SelectLowercase_ExperimentalStringFunctionsDisabled) {
  g_enable_experimental_string_functions = false;

  try {
    run_multiple_agg("select lower(first_name) from lower_function_test_people;",
                     ExecutorDeviceType::CPU);
    FAIL() << "An exception should have been thrown for this test case";
  } catch (const std::exception& e) {
    ASSERT_STREQ("Function LOWER(TEXT) not supported.", e.what());
    g_enable_experimental_string_functions = true;
  }
}

// end LOWER function tests

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  init();
  g_enable_experimental_string_functions = true;

  // Use system locale setting by default (as done in the server).
  boost::locale::generator generator;
  std::locale::global(generator.generate(""));

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_experimental_string_functions = false;
  reset();
  return err;
}
