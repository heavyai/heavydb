/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 *
 */

#include "Catalog/Catalog.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <boost/format.hpp>
#include <sstream>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace TestHelpers;

extern bool g_enable_string_functions;
extern unsigned g_trivial_loop_join_threshold;
extern bool g_enable_watchdog;
extern size_t g_watchdog_none_encoded_string_translation_limit;

namespace {

using QR = QueryRunner::QueryRunner;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

static const bool reuse_test_data = true;

inline auto sql(const std::string& sql_stmt,
                const ExecutorDeviceType device_type = ExecutorDeviceType::CPU,
                const bool enable_loop_joins = false) {
  const auto trivial_loop_join_threshold_state = g_trivial_loop_join_threshold;
  ScopeGuard reset_loop_join_state = [&trivial_loop_join_threshold_state] {
    g_trivial_loop_join_threshold = trivial_loop_join_threshold_state;
  };
  g_trivial_loop_join_threshold = 0;
  return QueryRunner::QueryRunner::get()->runSQL(
      sql_stmt, device_type, true, enable_loop_joins);
}

inline TargetValue run_simple_agg(const std::string& query_str,
                                  const ExecutorDeviceType device_type,
                                  const bool allow_loop_joins = true) {
  auto rows = QR::get()->runSQL(query_str, device_type, allow_loop_joins);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

inline auto multi_sql(const std::string& sql_stmts) {
  return QueryRunner::QueryRunner::get()
      ->runMultipleStatements(sql_stmts, ExecutorDeviceType::CPU)
      .back();
}

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

class AssertValueEqualsVisitor : public boost::static_visitor<> {
 public:
  AssertValueEqualsVisitor(const size_t& row, const size_t& column)
      : row(row), column(column) {}

  template <typename T, typename U>
  void operator()(const T& expected, const U& actual) const {
    if (std::is_pointer<U>::value) {
      EXPECT_EQ(1UL, 1UL);
    } else {
      FAIL() << boost::format(
                    "Values are of different types. Expected result set value: %s is of "
                    "type: %s while actual result set value: %s is of type: %s. At row: "
                    "%d, column: %d") %
                    expected % typeid(expected).name() % actual % typeid(actual).name() %
                    row % column;
    }
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

// begin string function tests

/**
 * @brief Class used for setting up and tearing down tables and records that are required
 * by the string function test cases
 */
class StringFunctionTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!reuse_test_data || !StringFunctionTest::test_data_loaded) {
      ASSERT_NO_THROW(multi_sql(R"(
          drop table if exists string_function_test_people;
          create table string_function_test_people(id int, first_name text, last_name text encoding none, full_name text, age integer, country_code text, us_phone_number text, zip_plus_4 text, personal_motto text, raw_email text);
          insert into string_function_test_people values(1, 'JOHN', 'SMITH', 'John SMITH', 25, 'us', '555-803-2144', '90210-7743', 'All for one and one for all.', 'Shoot me a note at therealjohnsmith@omnisci.com');
          insert into string_function_test_people values(2, 'John', 'Banks', 'John BANKS', 30, 'Us', '555-803-8244', '94104-8123', 'One plus one does not equal two.', 'Email: john_banks@mapd.com');
          insert into string_function_test_people values(3, 'JOHN', 'Wilson', 'John WILSON', 20, 'cA', '555-614-9814', null, 'What is the sound of one hand clapping?', 'JOHN.WILSON@geops.net');
          insert into string_function_test_people values(4, 'Sue', 'Smith', 'Sue SMITH', 25, 'CA', '555-614-2282', null, 'Nothing exists entirely alone. Everything is always in relation to everything else.', 'Find me at sue4tw@example.com, or reach me at sue.smith@example.com. I''d love to hear from you!'); 
          drop table if exists string_function_test_countries;
          create table string_function_test_countries(id int, code text, arrow_code text, name text, short_name text encoding none, capital text, largest_city text encoding none, lang text encoding none);
          insert into string_function_test_countries values(1, 'US', '>>US<<', 'United States', null, 'Washington', 'New York City', 'en');
          insert into string_function_test_countries values(2, 'ca', '>>CA<<', 'Canada', 'Canada', 'Ottawa', 'TORONTO', 'EN');
          insert into string_function_test_countries values(3, 'Gb', '>>GB<<', 'United Kingdom', 'UK', 'London', 'LONDON', 'en');
          insert into string_function_test_countries values(4, 'dE', '>>DE<<', 'Germany', 'Germany', 'Berlin', 'Berlin', 'de');
        )"));
      StringFunctionTest::test_data_loaded = true;
    }
  }

  void TearDown() override {
    if (!reuse_test_data) {
      ASSERT_NO_THROW(multi_sql(R"(
          drop table string_function_test_people;
          drop table string_function_test_countries;
        )"););
    }
  }
  static bool test_data_loaded;
};

bool StringFunctionTest::test_data_loaded = false;

TEST_F(StringFunctionTest, Lowercase) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select lower(first_name) from string_function_test_people order by id asc;", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"john"}, {"john"}, {"john"}, {"sue"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, LowercaseLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select lower('fUnNy CaSe');", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"funny case"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, Uppercase) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select upper(first_name) from string_function_test_people order by id asc;", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"JOHN"}, {"JOHN"}, {"JOHN"}, {"SUE"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, UppercaseLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select upper('fUnNy CaSe');", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"FUNNY CASE"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, InitCap) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select initcap(full_name) from string_function_test_people order by id asc", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"John Smith"}, {"John Banks"}, {"John Wilson"}, {"Sue Smith"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, InitCapLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select initcap('fUnNy CaSe');", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Funny Case"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, Reverse) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select reverse(full_name) from string_function_test_people order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"HTIMS nhoJ"}, {"SKNAB nhoJ"}, {"NOSLIW nhoJ"}, {"HTIMS euS"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, ReverseLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select reverse('fUnNy CaSe');", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"eSaC yNnUf"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, Repeat) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select repeat(full_name, 2) from string_function_test_people order by id asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"John SMITHJohn SMITH"},
        {"John BANKSJohn BANKS"},
        {"John WILSONJohn WILSON"},
        {"Sue SMITHSue SMITH"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, RepeatLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select repeat('fUnNy CaSe', 3);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"fUnNy CaSefUnNy CaSefUnNy CaSe"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, Concat) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select name || ', Earth' from string_function_test_countries order by id asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"United States, Earth"},
        {"Canada, Earth"},
        {"United Kingdom, Earth"},
        {"Germany, Earth"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, ReverseConcat) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select 'Country: ' || code from string_function_test_countries order by id asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"Country: US"}, {"Country: ca"}, {"Country: Gb"}, {"Country: dE"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, ConcatLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select 'fUnNy CaSe' || ' is the case.';", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"fUnNy CaSe is the case."}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, LPad) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select lpad(name, 14) from string_function_test_countries order by id asc;", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {" United States"}, {"        Canada"}, {"United Kingdom"}, {"       Germany"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, LPadTruncate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select lpad(name, 5) from string_function_test_countries order by id asc;", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"Unite"}, {"Canad"}, {"Unite"}, {"Germa"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, LPadCustomChars) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select lpad(name, 14, '>|<') from string_function_test_countries order by id "
        "asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {">United States"}, {">|<>|<>|Canada"}, {"United Kingdom"}, {">|<>|<>Germany"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, DISABLED_LPadLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select lpad('123', 2);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"  123"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, RPad) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select rpad(name, 20) from string_function_test_countries order by id asc;", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"United States       "},
        {"Canada              "},
        {"United Kingdom      "},
        {"Germany             "}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, RPadLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select rpad('$323.', 8, '98') from string_function_test_countries order by id "
        "asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"$323.989"}, {"$323.989"}, {"$323.989"}, {"$323.989"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, TrimBothDefault) {
  // Will be a no-op as default trim character is space
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select trim(arrow_code) from string_function_test_countries order by id asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {">>US<<"}, {">>CA<<"}, {">>GB<<"}, {">>DE<<"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, TrimBothCustom) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Implicit 'BOTH
    auto result_set1 =
        sql("select trim('<>' from arrow_code) from string_function_test_countries order "
            "by id asc;",
            dt);
    // explicit syntax
    auto result_set2 =
        sql("select trim(both '<>' from arrow_code) from string_function_test_countries "
            "order by id asc;",
            dt);

    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"US"}, {"CA"}, {"GB"}, {"DE"}};
    compare_result_set(expected_result_set, result_set1);
    compare_result_set(expected_result_set, result_set2);
  }
}

TEST_F(StringFunctionTest, TrimBothLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set1 = sql("select trim(both ' !' from ' Oops!');", dt);
    auto result_set2 = sql("select trim(' !' from ' Oops!');", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Oops"}};
    compare_result_set(expected_result_set, result_set1);
    compare_result_set(expected_result_set, result_set2);
  }
}

TEST_F(StringFunctionTest, LeftTrim) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // Trim with 'LEADING'
    auto result_set1 =
        sql("select trim(leading '<>#' from arrow_code) from "
            "string_function_test_countries order by id asc;",
            dt);

    // Explicit LTrim
    auto result_set2 = sql(
        "select ltrim(arrow_code, '<>#') from string_function_test_countries order by "
        "id asc;",
        dt);

    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"US<<"}, {"CA<<"}, {"GB<<"}, {"DE<<"}};

    compare_result_set(expected_result_set, result_set1);
    compare_result_set(expected_result_set, result_set2);
  }
}

TEST_F(StringFunctionTest, LeftTrimLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Trim with 'LEADING'
    auto result_set1 = sql("select trim(leading '$' from '$19.99$');", dt);

    // LTrim
    auto result_set2 = sql("select ltrim('$19.99$', '$');", dt);

    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"19.99$"}};

    compare_result_set(expected_result_set, result_set1);
    compare_result_set(expected_result_set, result_set2);
  }
}

TEST_F(StringFunctionTest, RightTrim) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Trim with 'TRAILING'
    auto result_set1 =
        sql("select trim(trailing '<> ' from arrow_code) from "
            "string_function_test_countries order by id asc;",
            dt);

    // RTrim
    auto result_set2 = sql(
        "select rtrim(arrow_code, '<> ') from string_function_test_countries order by "
        "id asc;",
        dt);

    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {">>US"}, {">>CA"}, {">>GB"}, {">>DE"}};

    compare_result_set(expected_result_set, result_set1);
    compare_result_set(expected_result_set, result_set2);
  }
}

TEST_F(StringFunctionTest, RightTrimLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Trim with 'TRAILING'
    auto result_set1 = sql("select trim(trailing '|' from '|half pipe||');");

    // RTrim
    auto result_set2 = sql("select rtrim('|half pipe||', '|');", dt);

    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"|half pipe"}};

    compare_result_set(expected_result_set, result_set1);
    compare_result_set(expected_result_set, result_set2);
  }
}

TEST_F(StringFunctionTest, Substring) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set1 = sql(
          "select substring(full_name, 1, 4) from string_function_test_people order by "
          "id asc;",
          dt);
      auto result_set2 =
          sql("select substring(full_name from 1 for 4) from string_function_test_people "
              "order by "
              "id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"John"}, {"John"}, {"John"}, {"Sue "}};
      compare_result_set(expected_result_set, result_set1);
      compare_result_set(expected_result_set, result_set2);
    }
    {
      // Test null inputs
      auto result_set1 = sql(
          "select substring(zip_plus_4, 1, 5) from string_function_test_people order by "
          "id asc;",
          dt);

      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"90210"}, {"94104"}, {""}, {""}};
    }
  }
}

TEST_F(StringFunctionTest, SubstringNegativeWrap) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select substring(full_name, -3, 2) from string_function_test_people order by "
        "id asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"IT"}, {"NK"}, {"SO"}, {"IT"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SubstringLengthOffEnd) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select substring(code, 2, 10) from string_function_test_countries order by id "
        "asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"S"}, {"a"}, {"b"}, {"E"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SubstringLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select substring('fUnNy CaSe', 4, 4);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Ny C"}};
    compare_result_set(expected_result_set, result_set);
  }
}

// Test that index of 0 is equivalent to index of 1 (first character)
TEST_F(StringFunctionTest, SubstringLengthZeroStartLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set1 = sql("select substring('12345', 1, 3);", dt);
    auto result_set2 = sql("select substring('12345', 0, 3);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"123"}};
    compare_result_set(expected_result_set, result_set1);
    compare_result_set(expected_result_set, result_set2);
  }
}

TEST_F(StringFunctionTest, SubstrAlias) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select substr(us_phone_number, 5, 3) from string_function_test_people order "
            "by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"803"}, {"803"}, {"614"}, {"614"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SubstrAliasLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select substr('fUnNy CaSe', 4, 4);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Ny C"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, Overlay) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select overlay(us_phone_number placing '6273' from 9) from "
            "string_function_test_people order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"555-803-6273"}, {"555-803-6273"}, {"555-614-6273"}, {"555-614-6273"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, OverlayInsert) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select overlay(us_phone_number placing '+1-' from 1 for 0) from "
            "string_function_test_people order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"+1-555-803-2144"},
                                                                    {"+1-555-803-8244"},
                                                                    {"+1-555-614-9814"},
                                                                    {"+1-555-614-2282"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, OverlayLiteralNoFor) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select overlay('We all love big data.' PLACING 'fast' FROM 13);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"We all love fastdata."}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, OverlayLiteralWithFor) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select overlay('We all love big data.' PLACING 'fast' FROM 13 FOR 3);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"We all love fast data."}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, Replace) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select replace(us_phone_number, '803', '#^!') from "
            "string_function_test_people order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"555-#^!-2144"}, {"555-#^!-8244"}, {"555-614-9814"}, {"555-614-2282"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, DISABLED_ReplaceEmptyReplacement) {
  // Todo: Determine why Calcite is not accepting 2-parameter version
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select replace(us_phone_number, '555-') from "
            "string_function_test_people order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"803-2144"}, {"803-8244"}, {"614-9814"}, {"614-2282"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, ReplaceLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select replace('We all love big data.', 'big', 'fast');", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"We all love fast data."}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, DISABLED_ReplaceLiteralEmptyReplacement) {
  // Todo: Determine why Calcite is not accepting 2-parameter version
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select replace('We all love big data.', 'big');", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"We all love data."}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SplitPart) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select split_part(us_phone_number, '-', 2) from string_function_test_people "
            "order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"803"}, {"803"}, {"614"}, {"614"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SplitPartNegativeIndex) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select split_part(us_phone_number, '-', -1) from "
            "string_function_test_people order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"2144"}, {"8244"}, {"9814"}, {"2282"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SplitPartNonDelimiterMatch) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set =
          sql("select split_part(name, ' ', -1) from "
              "string_function_test_countries order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"States"}, {"Canada"}, {"Kingdom"}, {"Germany"}};
      compare_result_set(expected_result_set, result_set);
    }

    for (int64_t split_idx = 0; split_idx <= 1; ++split_idx) {
      auto result_set = sql("select split_part(name, ' ', " + std::to_string(split_idx) +
                                ") from "
                                "string_function_test_countries order by id asc;",
                            dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"United"}, {"Canada"}, {"United"}, {"Germany"}};
      compare_result_set(expected_result_set, result_set);
    }
    {
      auto result_set =
          sql("select split_part(name, ' ', 2) from "
              "string_function_test_countries order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"States"}, {""}, {"Kingdom"}, {""}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, SplitPartLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select split_part('192.168.0.1', '.', 2);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"168"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SplitPartLiteralNegativeIndex) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select split_part('192.168.0.1', '.', -1);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"1"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SplitPartLiteralNullIndex) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql("select split_part('192.168.0.1', '.', 5);", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{""}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, RegexpReplace2Args) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select regexp_replace(name, 'United[[:space:]]') from "
            "string_function_test_countries order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"States"}, {"Canada"}, {"Kingdom"}, {"Germany"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, RegexpReplace3Args) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select regexp_replace(name, 'United[[:space:]]([[:alnum:]])', 'The United "
            "$1') from string_function_test_countries order by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"The United States"}, {"Canada"}, {"The United Kingdom"}, {"Germany"}};
    compare_result_set(expected_result_set, result_set);
  }
}

// 4th argument is position
TEST_F(StringFunctionTest, RegexpReplace4Args) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set =
          sql("select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
              "4) from string_function_test_people order by id asc");
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"All for one..two and one..two for all."},
          // Note we don't replace the first One due to start position argument of 4
          {"One plus one..two does not equal two."},
          {"What is the sound of one..two hand clapping?"},
          {"Nothing exists entirely alone. Everything is always in relation to "
           "everything else."}};
      compare_result_set(expected_result_set, result_set);
    }
    // Test negative position, should wrap
    {
      auto result_set =
          sql("select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
              "-18) from string_function_test_people order by id asc");
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"All for one and one..two for all."},
          // Note we don't replace the first One due to start position argument of 4
          {"One plus one does not equal two."},
          {"What is the sound of one..two hand clapping?"},
          {"Nothing exists entirely alone. Everything is always in relation to "
           "everything else."}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

// 5th argument is occurrence
TEST_F(StringFunctionTest, RegexpReplace5Args) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // 0 for 5th (occurrence) arguments says to replace all matches
    {
      auto result_set =
          sql("select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
              "1, 0) from string_function_test_people order by id asc");
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"All for one..two and one..two for all."},
          {"One..two plus one..two does not equal two."},
          {"What is the sound of one..two hand clapping?"},
          {"Nothing exists entirely alone. Everything is always in relation to "
           "everything else."}};
      compare_result_set(expected_result_set, result_set);
    }
    {
      // Replace second match
      auto result_set =
          sql("select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
              "1, 2) from string_function_test_people order by id asc");
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"All for one and one..two for all."},
          // Note we don't replace the first One due to start position argument of 4
          {"One plus one..two does not equal two."},
          {"What is the sound of one hand clapping?"},
          {"Nothing exists entirely alone. Everything is always in relation to "
           "everything else."}};
      compare_result_set(expected_result_set, result_set);
    }
    {
      // Replace second to last match via negative wrapping
      auto result_set =
          sql("select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
              "1, -2) from string_function_test_people order by id asc");
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"All for one..two and one for all."},
          // Note we don't replace the first One due to start position argument of 4
          {"One..two plus one does not equal two."},
          {"What is the sound of one hand clapping?"},
          {"Nothing exists entirely alone. Everything is always in relation to "
           "everything else."}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

// 6th argument is regex parameters
TEST_F(StringFunctionTest, RegexpReplace6Args) {
  // Currently only support 'c' (case sensitive-default) and 'i' (case insensitive) for
  // RegexpReplace
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Test 'c' - case sensitive
    {
      auto result_set =
          sql("select regexp_replace(personal_motto, '(one)[[:space:]]', '$1..two ', 1, "
              "0, 'c') from string_function_test_people order by id asc");
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"All for one..two and one..two for all."},
          // Note "One" in next entry doesn't match due to case sensitive search
          {"One plus one..two does not equal two."},
          {"What is the sound of one..two hand clapping?"},
          {"Nothing exists entirely alone. Everything is always in relation to "
           "everything else."}};
      compare_result_set(expected_result_set, result_set);
    }
    // Test 'i' - case insensitive
    {
      auto result_set =
          sql("select regexp_replace(personal_motto, '(one)[[:space:]]', '$1..two ', 1, "
              "0, 'i') from string_function_test_people order by id asc");
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"All for one..two and one..two for all."},
          // With case insensitive search, "One" will match
          {"One..two plus one..two does not equal two."},
          {"What is the sound of one..two hand clapping?"},
          {"Nothing exists entirely alone. Everything is always in relation to "
           "everything else."}};
      compare_result_set(expected_result_set, result_set);
    }
    // Test that invalid regex param causes exception
    {
      EXPECT_ANY_THROW(
          sql("select regexp_replace(personal_motto, '(one)[[:space:]]', '$1..two ', 1, "
              "0, 'iz') from string_function_test_people order by id asc;",
              dt));
    }
  }
}

TEST_F(StringFunctionTest, RegexpReplaceLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select regexp_replace('How much wood would a wood chuck chuck if a wood "
            "chuck could chuck wood?', 'wo[[:alnum:]]+d', 'metal', 1, 0, 'i');",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"How much metal metal a metal chuck chuck if a metal chuck could chuck metal?"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, RegexpReplaceLiteralSpecificMatch) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select regexp_replace('How much wood would a wood chuck chuck if a wood "
            "chuck could chuck wood?', 'wo[[:alnum:]]+d', 'should', 1, 2, 'i');",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"How much wood should a wood chuck chuck if a wood chuck could chuck wood?"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, RegexpSubstr2Args) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select regexp_substr(raw_email, '[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+') "
        "from string_function_test_people order by id asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"therealjohnsmith@omnisci.com"},
        {"john_banks@mapd.com"},
        {"JOHN.WILSON@geops.net"},
        {"sue4tw@example.com"}};
    compare_result_set(expected_result_set, result_set);
  }
}

// 3rd arg is start position
TEST_F(StringFunctionTest, RegexpSubstr3Args) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select regexp_substr(raw_email, '[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
        "20) from string_function_test_people order by id asc;",
        dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"therealjohnsmith@omnisci.com"}, {""}, {""}, {"sue.smith@example.com"}};
    compare_result_set(expected_result_set, result_set);
  }
}

// 4th arg is the occurence index
TEST_F(StringFunctionTest, RegexpSubstr4Args) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set = sql(
          "select regexp_substr(raw_email, '[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
          "1, 2) from string_function_test_people order by id asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {""}, {""}, {""}, {"sue.smith@example.com"}};
      compare_result_set(expected_result_set, result_set);
    }
    // Test negative wrapping
    {
      auto result_set = sql(
          "select regexp_substr(raw_email, '[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
          "1, -1) from string_function_test_people order by id asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"therealjohnsmith@omnisci.com"},
          {"john_banks@mapd.com"},
          {"JOHN.WILSON@geops.net"},
          {"sue.smith@example.com"}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

// 5th arg is regex params, 6th is sub-match index if 'e' is specified as regex param
TEST_F(StringFunctionTest, RegexpSubstr5Or6Args) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // case sensitive
    {
      auto result_set =
          sql("select regexp_substr(raw_email, "
              "'john[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', 1, 1, 'c') from "
              "string_function_test_people order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"johnsmith@omnisci.com"}, {"john_banks@mapd.com"}, {""}, {""}};
      compare_result_set(expected_result_set, result_set);
    }
    // case insensitive
    {
      auto result_set =
          sql("select regexp_substr(raw_email, "
              "'john[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', 1, 1, 'i') from "
              "string_function_test_people order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"johnsmith@omnisci.com"},
          {"john_banks@mapd.com"},
          {"JOHN.WILSON@geops.net"},
          {""}};
      compare_result_set(expected_result_set, result_set);
    }
    // extract sub-matches
    {
      // Get the email domain (second sub-match)
      auto result_set =
          sql("select regexp_substr(raw_email, "
              "'([[:alnum:]._-]+)@([[:alnum:]]+.[[:alnum:]]+)', 1, 1, 'ce', 2) from "
              "string_function_test_people order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"omnisci.com"}, {"mapd.com"}, {"geops.net"}, {"example.com"}};
      compare_result_set(expected_result_set, result_set);
    }
    {
      // Sub-match has no effect if extract ('e') is not specified
      auto result_set =
          sql("select regexp_substr(raw_email, "
              "'([[:alnum:]._-]+)@([[:alnum:]]+.[[:alnum:]]+)', 1, 1, 'i', 2) from "
              "string_function_test_people order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"therealjohnsmith@omnisci.com"},
          {"john_banks@mapd.com"},
          {"JOHN.WILSON@geops.net"},
          {"sue4tw@example.com"}};
      compare_result_set(expected_result_set, result_set);
    }
    // Throw error if regex param is not valid
    {
      EXPECT_ANY_THROW(
          sql("select regexp_substr(raw_email, "
              "'([[:alnum:]._-]+)@([[:alnum:]]+.[[:alnum:]]+)', 1, 1, 'z', 2) from "
              "string_function_test_people order by id asc;",
              dt));
    }
    // Throw error if case regex param not specified
    {
      EXPECT_ANY_THROW(
          sql("select regexp_substr(raw_email, "
              "'([[:alnum:]._-]+)@([[:alnum:]]+.[[:alnum:]]+)', 1, 1, 'e', 2) from "
              "string_function_test_people order by id asc;",
              dt));
    }
  }
}

TEST_F(StringFunctionTest, RegexpSubstrLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select regexp_substr('Feel free to send us an email at spam@devnull.com!', "
            "'[[:alnum:]]+@[[:alnum:]]+.[[:alnum:]]+',  1, -1, 'i', 0);",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"spam@devnull.com"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, StringFunctionEqualsFilterLHS) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set =
          sql("select first_name, last_name from string_function_test_people "
              "where lower(country_code) = 'us';",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"JOHN", "SMITH"},
                                                                      {"John", "Banks"}};
      compare_result_set(expected_result_set, result_set);
    }

    {
      auto result_set =
          sql("select COUNT(*) from string_function_test_people "
              "where initcap(first_name) = 'John';",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{int64_t(3)}};

      compare_result_set(expected_result_set, result_set);
    }

    {
      auto result_set =
          sql("select lower(first_name), first_name from string_function_test_people "
              "where upper('johN') = first_name;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"john", "JOHN"},
                                                                      {"john", "JOHN"}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, StringFunctionEqualsFilterRHS) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set =
          sql("select first_name, last_name from string_function_test_people "
              "where 'us' = lower(country_code);",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"JOHN", "SMITH"},
                                                                      {"John", "Banks"}};
      compare_result_set(expected_result_set, result_set);
    }

    {
      auto result_set =
          sql("select COUNT(*) from string_function_test_people "
              "where 'John' = initcap(first_name);",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{int64_t(3)}};

      compare_result_set(expected_result_set, result_set);
    }

    {
      auto result_set =
          sql("select lower(first_name), first_name from string_function_test_people "
              "where first_name = upper('johN');",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"john", "JOHN"},
                                                                      {"john", "JOHN"}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, StringFunctionFilterBothSides) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set =
          sql("select first_name, last_name from string_function_test_people "
              "where lower('US') = lower(country_code);",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"JOHN", "SMITH"},
                                                                      {"John", "Banks"}};
      compare_result_set(expected_result_set, result_set);
    }

    {
      auto result_set =
          sql("select COUNT(*) from string_function_test_people "
              "where initcap('joHN') = initcap(first_name);",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{int64_t(3)}};

      compare_result_set(expected_result_set, result_set);
    }

    {
      auto result_set =
          sql("select first_name, lower(first_name), first_name from "
              "string_function_test_people "
              "where upper(first_name) = upper('johN');",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN", "john", "JOHN"}, {"John", "john", "John"}, {"JOHN", "john", "JOHN"}};
      compare_result_set(expected_result_set, result_set);
    }

    {
      auto result_set =
          sql("select first_name, full_name from string_function_test_people "
              "where initcap(first_name) = split_part(full_name, ' ', 1);",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN", "John SMITH"},
          {"John", "John BANKS"},
          {"JOHN", "John WILSON"},
          {"Sue", "Sue SMITH"}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, MultipleFilters) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set =
          sql("select first_name, last_name from string_function_test_people "
              "where lower(country_code) = 'us' or lower(first_name) = 'sue';",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN", "SMITH"}, {"John", "Banks"}, {"Sue", "Smith"}};
      compare_result_set(expected_result_set, result_set);
    }
    {
      auto result_set =
          sql("select first_name, last_name from string_function_test_people "
              "where lower(country_code) = 'us' or upper(country_code) = 'CA';",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN", "SMITH"}, {"John", "Banks"}, {"JOHN", "Wilson"}, {"Sue", "Smith"}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, MixedFilters) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select first_name, last_name from string_function_test_people "
            "where lower(country_code) = 'ca' and age > 20;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Sue", "Smith"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, ChainedOperators) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set =
          sql("select initcap(split_part(full_name, ' ', 2)) as surname from "
              "string_function_test_people order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"Smith"}, {"Banks"}, {"Wilson"}, {"Smith"}};
      compare_result_set(expected_result_set, result_set);
    }
    {
      auto result_set =
          sql("select upper(split_part(split_part(regexp_substr(raw_email, "
              "'[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
              "1, -1), '@', -1), '.', 1)) as upper_domain from "
              "string_function_test_people order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"OMNISCI"}, {"MAPD"}, {"GEOPS"}, {"EXAMPLE"}};
      compare_result_set(expected_result_set, result_set);
    }
    {
      auto result_set =
          sql("select lower(split_part(split_part(regexp_substr(raw_email, "
              "'[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
              "1, -1), '@', -1), '.', 2)), "
              "upper(split_part(split_part(regexp_substr(raw_email, "
              "'[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
              "1, -1), '@', -1), '.', 1)) as upper_domain from "
              "string_function_test_people where substring(replace(raw_email, 'com', "
              "'org') from -3 for 3) = 'org' order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"com", "OMNISCI"},
                                                                      {"com", "MAPD"}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, CaseStatement) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Single column, string op only on output
    {
      auto result_set =
          sql("select case when first_name = 'JOHN' then lower(first_name) else "
              "upper(first_name) end "
              "as case_stmt from string_function_test_people order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"john"}, {"JOHN"}, {"john"}, {"SUE"}};
      compare_result_set(expected_result_set, result_set);
    }
    // Single column, string ops on inputs and outputs, with additional literal
    {
      auto result_set = sql(
          "select case when split_part(us_phone_number, '-', 2) = '614' then "
          "split_part(us_phone_number, '-', 3) "
          "when split_part(us_phone_number, '-', 3) = '2144' then "
          "substring(us_phone_number from 1 for 3) else "
          "'Surprise' end as case_stmt from string_function_test_people order by id asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"555"}, {"Surprise"}, {"9814"}, {"2282"}};
    }
    // Multi-column, string ops on inputs and outputs, with null and additional literal
    {
      auto result_set = sql(
          "select case when split_part(us_phone_number, '-', 2) = trim('614 ') then null "
          "when split_part(us_phone_number, '-', 3) = '214' || '4' then "
          "regexp_substr(zip_plus_4, "
          "'^[[:digit:]]+') else upper(country_code) end as case_stmt from "
          "string_function_test_people "
          "order by id asc;",
          dt);

      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"90210"}, {"US"}, {""}, {""}};
      compare_result_set(expected_result_set, result_set);
    }
    // [QE-359] Currently unsupported string op on output of case statement
    {
      EXPECT_ANY_THROW(
          sql("select upper(case when name like 'United%' then 'The ' || name "
              "else name end) from string_function_test_countries order by id asc;",
              dt));
    }
  }
}

TEST_F(StringFunctionTest, GroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      auto result_set =
          sql("select lower(first_name), count(*) from string_function_test_people "
              "group by lower(first_name) order by 2 desc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"john", int64_t(3)}, {"sue", int64_t(1)}};
      compare_result_set(expected_result_set, result_set);
    }
    {
      auto result_set = sql(
          "select regexp_substr(raw_email, "
          "'([[:alnum:]._-]+)@([[:alnum:]]+).([[:alnum:]]+)', 1, 1, 'ie', 3) as tld, "
          "count(*) as n from string_function_test_people group by tld order by tld asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"com", int64_t(3)}, {"net", int64_t(1)}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, Join) {
  // Turn off loop joins with the third parameter to ensure hash joins work
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Ensure loop join throws
    {
      EXPECT_ANY_THROW(
          sql("select COUNT(*) from string_function_test_people, "
              "string_function_test_countries;",
              dt,
              false));
    }
    // Both sides
    {
      auto result_set =
          sql("select first_name, name as country_name "
              "from string_function_test_people a "
              "join string_function_test_countries b on lower(country_code) = "
              "lower(code) order by a.id asc;",
              dt,
              false);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN", "United States"},
          {"John", "United States"},
          {"JOHN", "Canada"},
          {"Sue", "Canada"}};
      compare_result_set(expected_result_set, result_set);
    }
    // String op lhs
    {
      auto result_set =
          sql("select first_name, name as country_name "
              "from string_function_test_people a "
              "join string_function_test_countries b on upper(country_code) = code order "
              "by a.id asc;",
              dt,
              false);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN", "United States"}, {"John", "United States"}};
      compare_result_set(expected_result_set, result_set);
    }
    // String op rhs
    {
      auto result_set =
          sql("select first_name, name as country_name "
              "from string_function_test_people a "
              "join string_function_test_countries b on country_code = lower(code) order "
              "by a.id asc;",
              dt,
              false);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN", "United States"}};
      compare_result_set(expected_result_set, result_set);
    }
    // Two arg baseline join
    {
      auto result_set =
          sql("select first_name, upper(name) as upper_country_name "
              "from string_function_test_people a "
              "join string_function_test_countries b on lower(country_code) = "
              "lower(code) and upper(country_code) = repeat(code, 1) order by a.id asc;",
              dt,
              false);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN", "UNITED STATES"}, {"John", "UNITED STATES"}};
      compare_result_set(expected_result_set, result_set);
    }

    // [QE-359] Should throw when join predicate contains string op
    // on top of case statement
    {
      EXPECT_ANY_THROW(
          sql("select count(*) from string_function_test_people a inner join "
              "string_function_test_countries b on lower(code) = lower(case when "
              "code = 'US' then repeat(code, 2) else code end);",
              dt));
    }
  }
}

TEST_F(StringFunctionTest, SelectLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select first_name, lower('SMiTH') from string_function_test_people;", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"JOHN", "smith"}, {"John", "smith"}, {"JOHN", "smith"}, {"Sue", "smith"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, UpdateLowercase_EncodedColumnOnly) {
  auto result_set = multi_sql(R"(
      update string_function_test_people set country_code = lower(country_code);
      select country_code from string_function_test_people;
    )");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"us"}, {"us"}, {"ca"}, {"ca"}};
  compare_result_set(expected_result_set, result_set);
}

/**
 * UPDATE statements with at least one non-encoded column follow a different code path
 * from those with only encoded columns (see StorageIOFacility::yieldUpdateCallback for
 * more details).InsertIntoSelectLowercase
 */
TEST_F(StringFunctionTest, UpdateLowercase_EncodedAndNonEncodedColumns) {
  auto result_set = multi_sql(R"(
      update string_function_test_people set last_name = last_name, country_code = lower(country_code);
      select last_name, country_code from string_function_test_people;)");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"SMITH", "us"}, {"Banks", "us"}, {"Wilson", "ca"}, {"Smith", "ca"}};
  compare_result_set(expected_result_set, result_set);
}

// TODO: Re-enable after clear definition around handling non-ASCII characters
TEST_F(StringFunctionTest, DISABLED_LowercaseNonAscii) {
  sql("insert into string_function_test_people values('Ħ', 'Ħ', 25, 'GB')");
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select lower(first_name), last_name from string_function_test_people where "
            "country_code = 'GB';",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"ħ", "Ħ"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, LowercaseNoneEncodedTextColumn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set = sql(
        "select lower(last_name) from string_function_test_people order by id asc;", dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"smith"}, {"banks"}, {"wilson"}, {"smith"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, ChainNoneEncodedTextColumn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select reverse(initcap(last_name)) from string_function_test_people order "
            "by id asc;",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{
        {"htimS"}, {"sknaB"}, {"nosliW"}, {"htimS"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, NoneEncodedGroupByNoStringOps) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // No string ops
    {
      auto result_set =
          sql("select encode_text(lang) as g, count(*) as n from "
              "string_function_test_countries group by g order by g asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"de", int64_t(1)}, {"en", int64_t(2)}, {"EN", int64_t(1)}};
    }
  }
}

TEST_F(StringFunctionTest, NoneEncodedGroupByNullsNoStringOps) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // No string ops
    {
      auto result_set =
          sql("select encode_text(short_name) as g, count(*) as n from "
              "string_function_test_countries group by g order by g asc nulls last;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"Canada", int64_t(1)},
          {"Germany", int64_t(1)},
          {"UK", int64_t(1)},
          {"", int64_t(1)}};
    }
  }
}

TEST_F(StringFunctionTest, NoneEncodedGroupByStringOps) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // String ops
    {
      auto result_set =
          sql("select lower(lang) as g, count(*) as n from "
              "string_function_test_countries group by g order by n desc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"en", int64_t(3)},
                                                                      {"de", int64_t(1)}};
    }
    // With inert wrapping of ENCODE_TEXT
    {
      auto result_set =
          sql("select encode_text(lower(lang)) as g, count(*) as n from "
              "string_function_test_countries group by g order by n desc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"en", int64_t(3)},
                                                                      {"de", int64_t(1)}};
    }
    // With inner ENCODE_TEXT cast
    {
      auto result_set =
          sql("select lower(encode_text(lang)) as g, count(*) as n from "
              "string_function_test_countries group by g order by n desc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"en", int64_t(3)},
                                                                      {"de", int64_t(1)}};
    }

    {
      auto result_set =
          sql("select initcap(last_name) as g, count(*) as n from "
              "string_function_test_people group by g order by g asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"Banks", int64_t(1)}, {"Smith", int64_t(2)}, {"Wilson", int64_t(1)}};
    }

    // String ops with filter
    {
      auto result_set = sql(
          "select initcap(last_name) as g, count(*) as n from "
          "string_function_test_people where encode_text(last_name) <> upper(last_name) "
          "group by g order by g asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"Banks", int64_t(1)}, {"Smith", int64_t(1)}, {"Wilson", int64_t(1)}};
    }
  }
}

TEST_F(StringFunctionTest, NoneEncodedGroupByNullsStringOps) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // String ops
    {
      auto result_set =
          sql("select substring(short_name from 4 for 5) as g, count(*) as n from "
              "string_function_test_countries group by g order by g asc nulls last;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"ada", int64_t(1)}, {"many", int64_t(1)}, {"", int64_t(2)}};
    }
  }
}

TEST_F(StringFunctionTest, NoneEncodedEncodedEquality) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // None encoded = encoded, no string ops
    {
      auto result_set =
          sql("select name from string_function_test_countries where "
              "name = short_name order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Canada"},
                                                                      {"Germany"}};
      compare_result_set(expected_result_set, result_set);
    }
    // Encoded = none-encoded, no string ops
    {
      auto result_set =
          sql("select UPPER(short_name) from string_function_test_countries where "
              "short_name = name order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"CANADA"},
                                                                      {"GERMANY"}};
      compare_result_set(expected_result_set, result_set);
    }

    // None encoded = encoded, string ops both sides
    {
      auto result_set = sql(
          "select upper(last_name) from string_function_test_people where "
          "initcap(last_name) = split_part(initcap(full_name), ' ', 2) order by id asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"SMITH"}, {"BANKS"}, {"WILSON"}, {"SMITH"}};
      compare_result_set(expected_result_set, result_set);
    }
    // Encoded = none encoded, string ops both sides
    {
      auto result_set = sql(
          "select upper(last_name) from string_function_test_people where "
          "split_part(initcap(full_name), ' ', 2) = initcap(last_name) order by id asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"SMITH"}, {"BANKS"}, {"WILSON"}, {"SMITH"}};
      compare_result_set(expected_result_set, result_set);
    }

    // None encoded = encoded, string ops one side
    {
      auto result_set =
          sql("select repeat(largest_city, 2) from string_function_test_countries where "
              "initcap(largest_city) = capital order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"LONDONLONDON"},
                                                                      {"BerlinBerlin"}};
      compare_result_set(expected_result_set, result_set);
    }
    // Encoded = none encoded, string ops one side
    {
      auto result_set =
          sql("select substring(capital from 0 for 3) from "
              "string_function_test_countries where "
              "capital = initcap(largest_city) order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Lon"}, {"Ber"}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, NoneEncodedCaseStatementsNoStringOps) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // None-encoded + none-encoded, no string ops
    {
      // Note if we don't project out the id column we get a
      // ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED: Columnar conversion not
      // supported for variable length types error
      // Need to address this separately (precedes the string function work in
      // master)
      auto result_set =
          sql("select id, case when id <= 2 then short_name else lang end from "
              "string_function_test_countries order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {int64_t(1), ""},
          {int64_t(2), "Canada"},
          {int64_t(3), "en"},
          {int64_t(4), "de"}};
      compare_result_set(expected_result_set, result_set);
    }

    // None-encoded + none-encoded + literal
    {
      auto result_set = sql(
          "select id, case when id = 1 then 'USA' when id <= 3 then short_name else lang "
          "end from string_function_test_countries order by id asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {int64_t(1), "USA"},
          {int64_t(2), "Canada"},
          {int64_t(3), "UK"},
          {int64_t(4), "de"}};
      compare_result_set(expected_result_set, result_set);
    }

    // Dict-encoded + none-encoded + literal
    {
      auto result_set =
          sql("select case when id <= 2 then name when id <= 3 then short_name else 'DE' "
              "end from string_function_test_countries order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"United States"}, {"Canada"}, {"UK"}, {"DE"}};
      compare_result_set(expected_result_set, result_set);
    }

    // Group by
    // Dict-encoded + none-encoded + literal
    {
      auto result_set =
          sql("select case when lang = 'en' then lang when code = 'ca' then 'en' else "
              "code end "
              "as g, count(*) as n from string_function_test_countries group by g order "
              "by g asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"dE", int64_t(1)},
                                                                      {"en", int64_t(3)}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, NoneEncodedCaseStatementsStringOps) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // None-encoded + none-encoded, no string ops
    {
      auto result_set =
          sql("select case when id <= 2 then lower(short_name) else upper(lang) end from "
              "string_function_test_countries order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {""}, {"canada"}, {"EN"}, {"DE"}};
      compare_result_set(expected_result_set, result_set);
    }

    // None-encoded + none-encoded + literal
    {
      auto result_set = sql(
          "select case when id = 1 then initcap('USA') when id <= 3 then "
          "upper(short_name) "
          "else reverse(lang) end from string_function_test_countries order by id asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"Usa"}, {"CANADA"}, {"UK"}, {"ed"}};
      compare_result_set(expected_result_set, result_set);
    }

    // Dict-encoded + none-encoded + literal
    {
      auto result_set =
          sql("select case when id <= 2 then initcap(repeat(name, 2)) when id <= 3 then "
              "substring(short_name from 2 for 1) else 'DE' "
              "end from string_function_test_countries order by id asc;",
              dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"United Statesunited States"}, {"Canadacanada"}, {"K"}, {"DE"}};
      compare_result_set(expected_result_set, result_set);
    }

    // Group by
    // Dict-encoded + none-encoded + literal
    {
      auto result_set = sql(
          "select case when lang = 'en' then upper(lang) when code = 'ca' then 'en' else "
          "'Z' || trim(leading 'd' from repeat(code, 2)) end "
          "as g, count(*) as n from string_function_test_countries group by g order "
          "by g asc;",
          dt);
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"EN", int64_t(2)}, {"ZEdE", int64_t(1)}, {"en", int64_t(1)}};
      compare_result_set(expected_result_set, result_set);
    }
  }
}

TEST_F(StringFunctionTest, LowercaseNonTextColumn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    try {
      sql("select lower(age) from string_function_test_people;", dt);
      FAIL() << "An exception should have been thrown for this test case";
    } catch (const std::exception& e) {
      ASSERT_STREQ(
          "Cannot CAST from INTEGER to TEXT",
          //"Error instantiating LOWER operator. Expected text type for argument 1 "
          //"(operand).",
          e.what());
    }
  }
}

TEST_F(StringFunctionTest, LowercaseNullColumn) {
  sql("insert into string_function_test_people values(5, null, 'Empty', null, 25, 'US', "
      "'555-123-4567', '12345-8765', 'One.', 'null@nullbin.org');");
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto result_set =
        sql("select lower(first_name), last_name from string_function_test_people where "
            "last_name = 'Empty';",
            dt);
    std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"", "Empty"}};
    compare_result_set(expected_result_set, result_set);
  }
}

TEST_F(StringFunctionTest, SelectLowercase_StringFunctionsDisabled) {
  const auto previous_string_function_state = g_enable_string_functions;
  ScopeGuard reset_string_function_state = [&previous_string_function_state] {
    g_enable_string_functions = previous_string_function_state;
  };
  g_enable_string_functions = false;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    try {
      sql("select lower(first_name) from string_function_test_people;", dt);
      FAIL() << "An exception should have been thrown for this test case";
    } catch (const std::exception& e) {
      ASSERT_STREQ("Function LOWER not supported.", e.what());
    }
  }
}

TEST_F(StringFunctionTest, SelectLowercaseNoneEncoded_MoreRowsThanWatchdogLimit) {
  const auto previous_watchdog_state = g_enable_watchdog;
  const auto previous_none_encoded_translation_limit =
      g_watchdog_none_encoded_string_translation_limit;
  ScopeGuard reset_watchdog_state = [&previous_watchdog_state,
                                     &previous_none_encoded_translation_limit] {
    g_enable_watchdog = previous_watchdog_state;
    g_watchdog_none_encoded_string_translation_limit =
        previous_none_encoded_translation_limit;
  };
  g_enable_watchdog = true;
  g_watchdog_none_encoded_string_translation_limit = 3;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    try {
      sql("select lower(last_name) from string_function_test_people;", dt);
      FAIL() << "An exception should have been thrown for this test case";
    } catch (const std::exception& e) {
      std::ostringstream expected_error;
      expected_error
          << "Query requires one or more casts between none-encoded and "
             "dictionary-encoded "
          << "strings, and the estimated table size (5 rows) "
          << "exceeds the configured watchdog none-encoded string translation limit of "
          << g_watchdog_none_encoded_string_translation_limit << " rows.";
      const auto expected_error_str = expected_error.str();
      ASSERT_STREQ(expected_error_str.c_str(), e.what());
    }
  }
}

TEST_F(StringFunctionTest, UDF_ExpandDefaults) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    /*
    // ltrim has a defaulted second argument, test the parsing in group clause
    select ltrim(name) from heavyai_us_states;
    select ltrim(name) from heavyai_us_states group by 1;
    select ltrim(name) from heavyai_us_states group by name;
    select ltrim(name) as s from heavyai_us_states;
    select ltrim(name) as s from heavyai_us_states group by 1;
    select ltrim(name) as s from heavyai_us_states group by s;
    select ltrim(name) as s from heavyai_us_states group by name;

    // baseline verification
    select ltrim(name, 'New') from heavyai_us_states;
    select ltrim(name, 'New') from heavyai_us_states group by 1;
    select ltrim(name, 'New') from heavyai_us_states group by name;
    select ltrim(name, 'New') as s from heavyai_us_states;
    select ltrim(name, 'New') as s from heavyai_us_states group by 1;
    select ltrim(name, 'New') as s from heavyai_us_states group by s;
    select ltrim(name, 'New') as s from heavyai_us_states group by name;

    // note: "group by 1" causes problems with "order by"
    */

    {
      // Second argument is defaulted to ""
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"JOHN"}, {"JOHN"}, {"John"}, {"Sue"}};
      std::vector<std::vector<ScalarTargetValue>> expected_result_set_unique{
          {"JOHN"}, {"John"}, {"Sue"}};

      auto result_set1 =
          sql("select ltrim(first_name) from string_function_test_people where id < 5 "
              "order by first_name asc;",
              dt);
      compare_result_set(expected_result_set, result_set1);

      auto result_set2 =
          sql("select ltrim(first_name) from string_function_test_people where id < 5 "
              "group by 1;",
              dt);
      // just test the parsing ...
      // => an order by clause fails because of '1' substitution
      // => result set comparison could theorically fail due to ordering issues
      // compare_result_set(expected_result_set_unique, result_set2);

      auto result_set3 =
          sql("select ltrim(first_name) from string_function_test_people where id < 5 "
              "group by first_name order by first_name asc;",
              dt);
      compare_result_set(expected_result_set_unique, result_set3);

      auto result_set4 =
          sql("select ltrim(first_name) as s from string_function_test_people where id < "
              "5 order by first_name asc;",
              dt);
      compare_result_set(expected_result_set, result_set4);

      auto result_set5 =
          sql("select ltrim(first_name) as s from string_function_test_people where id < "
              "5 group by 1;",
              dt);
      // just test the parsing ...
      // => an order by clause fails because of '1' substitution
      // => result set comparison could theorically fail due to ordering issues
      // compare_result_set(expected_result_set_unique, result_set5);

      auto result_set6 =
          sql("select ltrim(first_name) as s from string_function_test_people where id < "
              "5 group by s order by ltrim(first_name) asc;",
              dt);
      compare_result_set(expected_result_set_unique, result_set6);

      auto result_set7 =
          sql("select ltrim(first_name) as s from string_function_test_people where id < "
              "5 group by first_name order by first_name asc;",
              dt);
      compare_result_set(expected_result_set_unique, result_set7);
    }

    {
      // fully specified call
      std::vector<std::vector<ScalarTargetValue>> expected_result_set{
          {"HN"}, {"ohn"}, {"HN"}, {"Sue"}};
      std::vector<std::vector<ScalarTargetValue>> expected_result_set_unique{
          {"HN"}, {"ohn"}, {"Sue"}};

      auto result_set1 =
          sql("select ltrim(first_name, 'JO') from string_function_test_people where id "
              "< 5 order by id asc;",
              dt);
      compare_result_set(expected_result_set, result_set1);

      auto result_set2 =
          sql("select ltrim(first_name, 'JO') from string_function_test_people where id "
              "< 5 group by 1;",
              dt);
      // just test the parsing ...
      // => an order by clause fails because of '1' substitution
      // => result set comparison could theorically fail due to ordering issues
      // compare_result_set(expected_result_set, result_set2);

      auto result_set3 =
          sql("select ltrim(first_name, 'JO') from string_function_test_people where id "
              "< 5 group by first_name order by first_name asc;",
              dt);
      compare_result_set(expected_result_set_unique, result_set3);

      auto result_set4 =
          sql("select ltrim(first_name, 'JO') as s from string_function_test_people "
              "where id < 5 order by id asc;",
              dt);
      compare_result_set(expected_result_set, result_set4);

      auto result_set5 =
          sql("select ltrim(first_name, 'JO') as s "
              "from string_function_test_people "
              "where id < 5 "
              "group by 1;",
              dt);
      // just test the parsing ...
      // => an order by clause fails because of '1' substitution
      // => result set comparison could theorically fail due to ordering issues
      // compare_result_set(expected_result_set, result_set5);

      auto result_set6 =
          sql("select ltrim(first_name, 'JO') as s "
              "from string_function_test_people "
              "where id < 5 "
              "group by s "
              "order by ltrim(first_name, 'JO') asc;",
              dt);
      // the grouping changes the intermediate results, so use a special result set
      std::vector<std::vector<ScalarTargetValue>> expected_result_set_special{
          {"HN"}, {"Sue"}, {"ohn"}};
      compare_result_set(expected_result_set_special, result_set6);

      auto result_set7 =
          sql("select ltrim(first_name, 'JO') as s "
              "from string_function_test_people "
              "where id < 5 "
              "group by first_name "
              "order by first_name asc;",
              dt);
      compare_result_set(expected_result_set_unique, result_set7);
    }
  }
}

const char* postgres_osm_names = R"(
    CREATE TABLE postgres_osm_names (
      name TEXT,
      name_none_encoded TEXT ENCODING NONE,
      name_lower TEXT,
      name_upper TEXT,
      name_initcap TEXT,
      name_reverse TEXT,
      name_repeat_2 TEXT,
      name_concat_comma TEXT,
      comma_concat_name TEXT,
      name_lpad_20 TEXT,
      name_rpad_20 TEXT,
      name_trib_abab TEXT,
      name_ltrim_abab TEXT,
      name_rtrim_abab TEXT,
      name_substring_from_3_for_8 TEXT,
      name_overlay_foobar_from_4_for_3 TEXT,
      name_replace_il_eel TEXT,
      name_split_part_space_2 TEXT);
    )";

// Todo(todd): Add more Postgres tests. Part of the issue is just
// getting in the data correctly, for example, when we import anything
// with leading or trailing spaces (i.e. the 'name_substring_from_3_for_8
// column), our importer automatically drops the spaces, and there seems
// no easy way in Postgres to force quoting everything.
class PostgresStringFunctionTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!reuse_test_data || !PostgresStringFunctionTest::test_data_loaded) {
      ASSERT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS postgres_osm_names;"));
      ASSERT_NO_THROW(run_ddl_statement(postgres_osm_names));
      const std::string import_file{"postgres_osm_names_ascii_1k.csv.gz"};
      const auto load_str =
          std::string("COPY " + PostgresStringFunctionTest::table_name_ + " FROM '" +
                      "../../Tests/Import/datafiles/string_funcs/" + import_file +
                      "' WITH (header='true');");
      ASSERT_NO_THROW(run_ddl_statement(load_str));
      const int64_t row_count = v<int64_t>(run_simple_agg(
          "SELECT COUNT(*) FROM " + PostgresStringFunctionTest::table_name_ + ";",
          ExecutorDeviceType::CPU));
      EXPECT_EQ(row_count, PostgresStringFunctionTest::expected_row_count_);
      PostgresStringFunctionTest::test_data_loaded = true;
    }
  }
  void TearDown() override {
    if (!reuse_test_data) {
      ASSERT_NO_THROW(run_ddl_statement("DROP TABLE postgres_osm_names;"));
    }
  }

  static const std::string table_name_;
  static const std::string orig_encoded_string_col_;
  static const std::string orig_none_encoded_string_col_;
  static const int64_t expected_row_count_;
  static bool test_data_loaded;
};

const std::string PostgresStringFunctionTest::table_name_ = "postgres_osm_names";
const std::string PostgresStringFunctionTest::orig_encoded_string_col_ = "name";
const std::string PostgresStringFunctionTest::orig_none_encoded_string_col_ =
    "name_none_encoded";
const int64_t PostgresStringFunctionTest::expected_row_count_ = 1000L;

bool PostgresStringFunctionTest::test_data_loaded = false;

TEST_F(PostgresStringFunctionTest, Count) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_EQ(expected_row_count_,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM " + PostgresStringFunctionTest::table_name_ + ";",
                  dt)));
  }
}

TEST_F(PostgresStringFunctionTest, Lower) {
  for (auto string_col : {PostgresStringFunctionTest::orig_encoded_string_col_,
                          PostgresStringFunctionTest::orig_none_encoded_string_col_}) {
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      EXPECT_EQ(expected_row_count_,
                v<int64_t>(run_simple_agg(
                    "SELECT COUNT(*) FROM " + PostgresStringFunctionTest::table_name_ +
                        " WHERE LOWER(" + string_col + ") = name_lower;",
                    dt)));
      EXPECT_EQ(0L,
                v<int64_t>(run_simple_agg(
                    "SELECT COUNT(*) FROM " + PostgresStringFunctionTest::table_name_ +
                        " WHERE LOWER(" + string_col + ") <> name_lower;",
                    dt)));
    }
  }
}

TEST_F(PostgresStringFunctionTest, Upper) {
  for (auto string_col : {PostgresStringFunctionTest::orig_encoded_string_col_,
                          PostgresStringFunctionTest::orig_none_encoded_string_col_}) {
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      EXPECT_EQ(expected_row_count_,
                v<int64_t>(run_simple_agg(
                    "SELECT COUNT(*) FROM " + PostgresStringFunctionTest::table_name_ +
                        " WHERE UPPER(" + string_col + ") = name_upper;",
                    dt)));
      EXPECT_EQ(0L,
                v<int64_t>(run_simple_agg(
                    "SELECT COUNT(*) FROM " + PostgresStringFunctionTest::table_name_ +
                        " WHERE UPPER(" + string_col + ") <> name_upper;",
                    dt)));
    }
  }
}

TEST_F(PostgresStringFunctionTest, InitCap) {
  for (auto string_col : {PostgresStringFunctionTest::orig_encoded_string_col_,
                          PostgresStringFunctionTest::orig_none_encoded_string_col_}) {
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      // Postgres seems to have different rules for INITCAP than the ones we use following
      // Snowflake, such as capitalizing letters after apostrophes (i.e. Smith'S), so
      // so exclude these differences via additional SQL filters
      EXPECT_EQ(
          0L,
          v<int64_t>(run_simple_agg(
              "SELECT (SELECT COUNT(*) FROM " + PostgresStringFunctionTest::table_name_ +
                  " where not name ilike '%''%' and not name ilike '%’%' and not name "
                  "ilike '%–%') "
                  " - (SELECT COUNT(*) FROM " +
                  PostgresStringFunctionTest::table_name_ + " WHERE INITCAP(" +
                  string_col +
                  ") = name_initcap AND not name ilike '%''%' "
                  " and not name ilike '%’%' and not name ilike '%–%');",
              dt)));
    }
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  QueryRunner::QueryRunner::init(BASE_PATH);
  g_enable_string_functions = true;
  g_enable_watchdog = true;
  g_watchdog_none_encoded_string_translation_limit = 1000000UL;

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_string_functions = false;
  QueryRunner::QueryRunner::reset();
  return err;
}
