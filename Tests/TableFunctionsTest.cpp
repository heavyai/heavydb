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

#include "ArrowSQLRunner.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace TestHelpers;
using namespace TestHelpers::ArrowSQLRunner;

extern bool g_enable_table_functions;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !gpusPresent();
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

class TableFunctions : public ::testing::Test {
  void SetUp() override {
    {
      createTable("tf_test",
                  {{"x", SQLTypeInfo(kINT)},
                   {"f", SQLTypeInfo(kFLOAT)},
                   {"d", SQLTypeInfo(kDOUBLE)},
                   {"d2", SQLTypeInfo(kDOUBLE)}},
                  {2});

      std::stringstream ss;
      for (int i = 0; i < 5; i++) {
        ss << i << "," << (i * 1.1) << "," << (i * 1.1) << "," << (1.0 - i * 2.2)
           << std::endl;
      }
      insertCsvValues("tf_test", ss.str());
    }
    {
      createTable(
          "sd_test",
          {{"base", dictType(4, false, -1)}, {"derived", dictType(4, false, -1)}});
      insertCsvValues("sd_test", "hello,world\nfoo,bar\nbar,baz\nworld,foo\nbaz,hello");
    }
    {
      createTable("err_test",
                  {{"x", SQLTypeInfo(kINT)},
                   {"y", SQLTypeInfo(kBIGINT)},
                   {"f", SQLTypeInfo(kFLOAT)},
                   {"d", SQLTypeInfo(kDOUBLE)},
                   {"x2", SQLTypeInfo(kINT)}},
                  {2});

      std::stringstream ss;
      for (int i = 0; i < 5; i++) {
        ss << (std::numeric_limits<int32_t>::max() - 1) << ","
           << (std::numeric_limits<int64_t>::max() - 1) << ","
           << (std::numeric_limits<float>::max() - 1.0) << ","
           << (std::numeric_limits<double>::max() - 1.0) << "," << i << std::endl;
      }
      insertCsvValues("err_test", ss.str());
    }
  }

  void TearDown() override {
    dropTable("tf_test");
    dropTable("sd_test");
    dropTable("err_test");
  }
};

TEST_F(TableFunctions, BasicProjection) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), 0)) ORDER BY "
          "out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(0));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), 1)) ORDER BY "
          "out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), 2)) ORDER BY "
          "out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(10));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), 3)) ORDER BY "
          "out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(15));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), 4)) ORDER BY "
          "out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(20));
    }
    if (dt == ExecutorDeviceType::CPU) {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier2(cursor(SELECT d FROM tf_test), 0)) ORDER "
          "BY "
          "out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(0));
    }
    if (dt == ExecutorDeviceType::CPU) {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier2(cursor(SELECT d FROM tf_test), 1)) ORDER "
          "BY "
          "out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_adder(1, cursor(SELECT d, d2 FROM tf_test)));", dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_adder(4, cursor(SELECT d, d2 FROM tf_test)));", dt);
      ASSERT_EQ(rows->rowCount(), size_t(20));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0, out1 FROM TABLE(row_addsub(1, cursor(SELECT d, d2 FROM "
          "tf_test)));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    // Omit sizer (kRowMultiplier)
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_adder(cursor(SELECT d, d2 FROM tf_test)));", dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test))) ORDER BY "
          "out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    // Constant (kConstant) size tests with get_max_with_row_offset
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(get_max_with_row_offset(cursor(SELECT x FROM "
          "tf_test)));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]),
                static_cast<int64_t>(4));  // max value of x
    }
    {
      // swap output column order
      const auto rows = run_multiple_agg(
          "SELECT out1, out0 FROM TABLE(get_max_with_row_offset(cursor(SELECT x FROM "
          "tf_test)));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]),
                static_cast<int64_t>(4));  // row offset of max x
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]),
                static_cast<int64_t>(4));  // max value of x
    }
    // Table Function specified sizer test
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(column_list_row_sum(cursor(SELECT x, x FROM "
          "tf_test)));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(2));
    }
    // TextEncodingDict specific tests
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier_text(cursor(SELECT base FROM sd_test),"
          "1));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
      std::vector<std::string> expected_result_set{"hello", "foo", "bar", "world", "baz"};
      for (size_t i = 0; i < 5; i++) {
        auto row = rows->getNextRow(true, false);
        auto s = boost::get<std::string>(TestHelpers::v<NullableString>(row[0]));
        ASSERT_EQ(s, expected_result_set[i]);
      }
    }
    {
      const auto rows = run_multiple_agg("SELECT base FROM sd_test;", dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
      std::vector<std::string> expected_result_set{"hello", "foo", "bar", "world", "baz"};
      for (size_t i = 0; i < 5; i++) {
        auto row = rows->getNextRow(true, false);
        auto s = boost::get<std::string>(TestHelpers::v<NullableString>(row[0]));
        ASSERT_EQ(s, expected_result_set[i]);
      }
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier_text(cursor(SELECT derived FROM sd_test),"
          "1));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
      std::vector<std::string> expected_result_set{"world", "bar", "baz", "foo", "hello"};
      for (size_t i = 0; i < 5; i++) {
        auto row = rows->getNextRow(true, false);
        auto s = boost::get<std::string>(TestHelpers::v<NullableString>(row[0]));
        ASSERT_EQ(s, expected_result_set[i]);
      }
    }

    // Test boolean scalars AND return of less rows than allocated in table function
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(sort_column_limit(CURSOR(SELECT x FROM tf_test), 2, "
          "true, "
          "true)) ORDER by out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(2));
      std::vector<int64_t> expected_result_set{0, 1};
      for (size_t i = 0; i < 2; i++) {
        auto row = rows->getNextRow(true, false);
        auto v = TestHelpers::v<int64_t>(row[0]);
        ASSERT_EQ(v, expected_result_set[i]);
      }
    }

    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(sort_column_limit(CURSOR(SELECT x FROM tf_test), 3, "
          "false, "
          "true)) ORDER by out0 DESC;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(3));
      std::vector<int64_t> expected_result_set{4, 3, 2};
      for (size_t i = 0; i < 3; i++) {
        auto row = rows->getNextRow(true, false);
        auto v = TestHelpers::v<int64_t>(row[0]);
        ASSERT_EQ(v, expected_result_set[i]);
      }
    }

    // Test for columns containing null values (QE-163)
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_test_nullable(cursor(SELECT x from tf_test), 1)) "
          "where out0 is not null;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(2));
      std::vector<int64_t> expected_result_set{1, 3};
      for (size_t i = 0; i < 2; i++) {
        auto row = rows->getNextRow(true, false);
        auto v = TestHelpers::v<int64_t>(row[0]);
        ASSERT_EQ(v, expected_result_set[i]);
      }
    }

    // Tests various invalid returns from a table function:
    if (dt == ExecutorDeviceType::CPU) {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier2(cursor(SELECT d FROM tf_test), -1));", dt);
      ASSERT_EQ(rows->rowCount(), size_t(0));
    }

    if (dt == ExecutorDeviceType::CPU) {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(row_copier2(cursor(SELECT d FROM tf_test), -2));",
              dt),
          std::runtime_error);
    }

    // TODO: enable the following tests after QE-50 is resolved:
    if (false && dt == ExecutorDeviceType::CPU) {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(row_copier2(cursor(SELECT d FROM tf_test), -3));",
              dt),
          std::runtime_error);
    }

    if (false && dt == ExecutorDeviceType::CPU) {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(row_copier2(cursor(SELECT d FROM tf_test), -4));",
              dt),
          std::runtime_error);
    }

    if (false && dt == ExecutorDeviceType::CPU) {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(row_copier2(cursor(SELECT d FROM tf_test), -5));",
              dt),
          std::runtime_error);
    }
  }
}

TEST_F(TableFunctions, GroupByIn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test GROUP BY d), "
          "1)) "
          "ORDER BY out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test GROUP BY d), "
          "2)) "
          "ORDER BY out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(10));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test GROUP BY d), "
          "3)) "
          "ORDER BY out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(15));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM tf_test GROUP BY d), "
          "4)) "
          "ORDER BY out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(20));
    }
  }
}

TEST_F(TableFunctions, GroupByInAndOut) {
  auto check_result = [](const auto rows, const size_t copies) {
    ASSERT_EQ(rows->rowCount(), size_t(5));
    for (size_t i = 0; i < 5; i++) {
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]), static_cast<int64_t>(copies));
    }
  };

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg(
          "SELECT out0, count(*) FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), "
          "1)) "
          "GROUP BY out0 ORDER BY out0;",
          dt);
      check_result(rows, 1);
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0, count(*) FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), "
          "2)) "
          "GROUP BY out0 ORDER BY out0;",
          dt);
      check_result(rows, 2);
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0, count(*) FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), "
          "3)) "
          "GROUP BY out0 ORDER BY out0;",
          dt);
      check_result(rows, 3);
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0, count(*) FROM TABLE(row_copier(cursor(SELECT d FROM tf_test), "
          "4)) "
          "GROUP BY out0 ORDER BY out0;",
          dt);
      check_result(rows, 4);
    }
    // TextEncodingDict specific tests
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier_text(cursor(SELECT base FROM sd_test),"
          "1)) "
          "ORDER BY out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
      std::vector<std::string> expected_result_set{"bar", "baz", "foo", "hello", "world"};
      for (size_t i = 0; i < 5; i++) {
        auto row = rows->getNextRow(true, false);
        auto s = boost::get<std::string>(TestHelpers::v<NullableString>(row[0]));
        ASSERT_EQ(s, expected_result_set[i]);
      }
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(row_copier_text(cursor(SELECT derived FROM sd_test),"
          "1)) "
          "ORDER BY out0;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
      std::vector<std::string> expected_result_set{"bar", "baz", "foo", "hello", "world"};
      for (size_t i = 0; i < 5; i++) {
        auto row = rows->getNextRow(true, false);
        auto s = boost::get<std::string>(TestHelpers::v<NullableString>(row[0]));
        ASSERT_EQ(s, expected_result_set[i]);
      }
    }
  }
}

TEST_F(TableFunctions, ConstantCasts) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Numeric constant to float
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_scalar_multiply(CURSOR(SELECT f FROM "
          "tf_test), 2.2));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    // Numeric constant to double
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_scalar_multiply(CURSOR(SELECT d FROM "
          "tf_test), 2.2));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    // Integer constant to double
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_scalar_multiply(CURSOR(SELECT d FROM "
          "tf_test), 2));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    // Numeric (integer) constant to double
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_scalar_multiply(CURSOR(SELECT d FROM "
          "tf_test), 2.));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    // Integer constant
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_scalar_multiply(CURSOR(SELECT x FROM "
          "tf_test), 2));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    // Should throw: Numeric constant to integer
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(ct_binding_scalar_multiply(CURSOR(SELECT x FROM "
              "tf_test), 2.2));",
              dt),
          std::exception);
    }
    // Should throw: boolean constant to integer
    {
      EXPECT_THROW(run_multiple_agg(
                       "SELECT out0 FROM TABLE(ct_binding_scalar_multiply(CURSOR(SELECT "
                       "x FROM tf_test), true));",
                       dt),
                   std::invalid_argument);
    }
  }
}

TEST_F(TableFunctions, Template) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_column2(cursor(SELECT x FROM tf_test), "
          "cursor(SELECT d from tf_test)))",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(10));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_column2(cursor(SELECT d FROM tf_test), "
          "cursor(SELECT d2 from tf_test)))",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(20));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_column2(cursor(SELECT x FROM tf_test), "
          "cursor(SELECT x from tf_test)))",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(30));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_column2(cursor(SELECT d FROM tf_test), "
          "cursor(SELECT x from tf_test)))",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(40));
    }
    // TextEncodingDict
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_column2(cursor(SELECT base FROM sd_test),"
          "cursor(SELECT derived from sd_test)))",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
      std::vector<std::string> expected_result_set{"hello", "foo", "bar", "world", "baz"};
      for (size_t i = 0; i < 5; i++) {
        auto row = rows->getNextRow(true, false);
        auto s = boost::get<std::string>(TestHelpers::v<NullableString>(row[0]));
        ASSERT_EQ(s, expected_result_set[i]);
      }
    }
  }
}

TEST_F(TableFunctions, Unsupported) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    EXPECT_THROW(run_multiple_agg("select * from table(row_copier(cursor(SELECT d, "
                                  "cast(x as double) FROM tf_test), 2));",
                                  dt),
                 std::runtime_error);
  }
}

TEST_F(TableFunctions, CallFailure) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    EXPECT_THROW(run_multiple_agg("SELECT out0 FROM TABLE(row_copier(cursor("
                                  "SELECT d FROM tf_test),101));",
                                  dt),
                 std::runtime_error);

    // Skip this test for GPU. TODO: row_copier return value is ignored.
    break;
  }
}

TEST_F(TableFunctions, NamedOutput) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg(
          "SELECT total FROM TABLE(ct_named_output(cursor(SELECT d FROM tf_test)));", dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<double>(crt_row[0]), static_cast<double>(11));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT total FROM TABLE(ct_named_const_output(cursor(SELECT x FROM "
          "tf_test)));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(2));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(6));
      crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(4));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT total FROM TABLE(ct_named_user_const_output(cursor(SELECT x FROM "
          "tf_test), 1));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(10));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT total FROM TABLE(ct_named_user_const_output(cursor(SELECT x FROM "
          "tf_test), 2));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(2));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(6));
      crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(4));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT total FROM TABLE(ct_named_rowmul_output(cursor(SELECT x FROM "
          "tf_test), 1));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT total FROM TABLE(ct_named_rowmul_output(cursor(SELECT x FROM "
          "tf_test), 2));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(10));
    }
  }
}

TEST_F(TableFunctions, CursorlessInputs) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg(
          "SELECT answer FROM TABLE(ct_no_arg_constant_sizing()) ORDER BY answer;", dt);
      ASSERT_EQ(rows->rowCount(), size_t(42));
      for (size_t i = 0; i < 42; i++) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(42 * i));
      }
    }

    {
      const auto rows = run_multiple_agg(
          "SELECT answer / 882 AS g, COUNT(*) AS n FROM "
          "TABLE(ct_no_arg_constant_sizing()) GROUP BY g ORDER BY g;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(2));

      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(0));
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]), static_cast<int64_t>(21));

      crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(1));
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]), static_cast<int64_t>(21));
    }

    {
      const auto rows =
          run_multiple_agg("SELECT answer FROM TABLE(ct_no_arg_runtime_sizing());", dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(42));
    }

    {
      const auto rows = run_multiple_agg(
          "SELECT answer FROM TABLE(ct_scalar_1_arg_runtime_sizing(123));", dt);
      ASSERT_EQ(rows->rowCount(), size_t(3));

      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(123));
      crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(12));
      crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(1));
    }

    {
      const auto rows = run_multiple_agg(
          "SELECT answer1, answer2 FROM TABLE(ct_scalar_2_args_constant_sizing(100, 5));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));

      for (size_t r = 0; r < 5; ++r) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(100 + r * 5));
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]), static_cast<int64_t>(100 - r * 5));
      }
    }

    // Tests for user-defined constant parameter sizing, which were separately broken from
    // the above
    {
      const auto rows = run_multiple_agg(
          "SELECT output FROM TABLE(ct_no_cursor_user_constant_sizer(8, 10));", dt);
      ASSERT_EQ(rows->rowCount(), size_t(10));

      for (size_t r = 0; r < 10; ++r) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(8));
      }
    }

    {
      const auto rows = run_multiple_agg(
          "SELECT output FROM TABLE(ct_templated_no_cursor_user_constant_sizer(7, 4));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(4));

      for (size_t r = 0; r < 4; ++r) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(7));
      }
    }

    {
      // Mandelbrot table function requires max_iterations to be >= 1
      const auto rows = run_multiple_agg(
          "SELECT MIN(num_iterations) AS min_iterations, MAX(num_iterations) AS "
          "max_iterations, COUNT(*) AS n FROM TABLE(tvf_mandelbrot(128 /* width */ , 128 "
          "/* height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y */, 1.0 /* "
          "max_y */, 256 /* max_iterations */));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]),
                static_cast<int64_t>(1));  // min_iterations
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]),
                static_cast<int64_t>(256));  // max_iterations
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[2]),
                static_cast<int64_t>(16384));  // num pixels - width X height
    }
  }
}

TEST_F(TableFunctions, TextEncodedNoneLiteralArgs) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Following tests ability to transform to std::string running on CPU (runs on CPU
    // only)
    {
      const std::string test_string{"this is only a test"};
      const size_t test_string_len{test_string.size()};
      const std::string test_query(
          "SELECT char_idx, char_bytes FROM TABLE(ct_string_to_chars('" + test_string +
          "')) ORDER BY char_idx;");
      const auto rows = run_multiple_agg(test_query, dt);
      ASSERT_EQ(rows->rowCount(), test_string_len);
      for (size_t r = 0; r < test_string_len; ++r) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(r));
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]),
                  static_cast<int64_t>(test_string[r]));
      }
    }
    // Following tests two text encoding none input, plus running on GPU + CPU
    {
      const std::string test_string1{"theater"};
      const std::string test_string2{"theatre"};
      const std::string test_query(
          "SELECT hamming_distance FROM TABLE(ct_hamming_distance('" + test_string1 +
          "','" + test_string2 + "'));");
      const auto rows = run_multiple_agg(test_query, dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(2));
    }

    // Following tests varchar element accessors and that TextEncodedNone literal inputs
    // play nicely with column inputs + RowMultiplier
    {
      const std::string test_string{"theater"};
      const std::string test_query(
          "SELECT idx, char_bytes FROM TABLE(ct_get_string_chars(CURSOR(SELECT x FROM "
          "tf_test), '" +
          test_string + "', 1)) ORDER BY idx;");
      const auto rows = run_multiple_agg(test_query, dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));  // size of tf_test
      for (size_t r = 0; r < 5; ++r) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(r));
        ASSERT_EQ(
            TestHelpers::v<int64_t>(crt_row[1]),
            static_cast<int64_t>(test_string[r]));  // x in tf_test is {1, 2, 3, 4, 5}
      }
    }
  }
}

TEST_F(TableFunctions, ThrowingTests) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(column_list_safe_row_sum(cursor(SELECT x FROM "
              "err_test)));",
              dt),
          std::runtime_error);
    }
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(column_list_safe_row_sum(cursor(SELECT y FROM "
              "err_test)));",
              dt),
          std::runtime_error);
    }
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(column_list_safe_row_sum(cursor(SELECT f FROM "
              "err_test)));",
              dt),
          std::runtime_error);
    }
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT out0 FROM TABLE(column_list_safe_row_sum(cursor(SELECT d FROM "
              "err_test)));",
              dt),
          std::runtime_error);
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(column_list_safe_row_sum(cursor(SELECT x2 FROM "
          "err_test)));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]),
                static_cast<int32_t>(10));  // 0+1+2+3+4=10
    }
    {
      // Mandelbrot table function requires max_iterations to be >= 1
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(tvf_mandelbrot(128 /* width */ , 128 /* "
                           "height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y "
                           "*/, 1.0 /* max_y */, 0 /* max_iterations */));",
                           dt),
          std::runtime_error);
    }

    // Ensure TableFunctionMgr and error throwing works properly for templated CPU TVFs
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT * FROM TABLE(ct_throw_if_gt_100(CURSOR(SELECT CAST(f AS FLOAT) AS "
              "f FROM (VALUES (0.0), (1.0), (2.0), (110.0)) AS t(f))));",
              dt),
          std::runtime_error);
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT CAST(val AS INT) AS val FROM TABLE(ct_throw_if_gt_100(CURSOR(SELECT "
          "CAST(f AS DOUBLE) AS f FROM (VALUES (0.0), (1.0), (2.0), (3.0)) AS t(f)))) "
          "ORDER BY val;",
          dt);
      const size_t num_rows = rows->rowCount();
      ASSERT_EQ(num_rows, size_t(4));
      for (size_t r = 0; r < num_rows; ++r) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(r));
      }
    }
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  // Table function support must be enabled before initialized the query runner
  // environment
  g_enable_table_functions = true;
  init();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  reset();
  return err;
}
