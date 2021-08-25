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

#include "TestHelpers.h"

#include <gtest/gtest.h>

#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

extern bool g_enable_table_functions;
namespace {

inline void run_ddl_statement(const std::string& stmt) {
  QR::get()->runDDLStatement(stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, false, false);
}

}  // namespace

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

class TableFunctions : public ::testing::Test {
  void SetUp() override {
    {
      run_ddl_statement("DROP TABLE IF EXISTS tf_test;");
      run_ddl_statement(
          "CREATE TABLE tf_test (x INT, f FLOAT, d DOUBLE, d2 DOUBLE) WITH "
          "(FRAGMENT_SIZE=2);");

      TestHelpers::ValuesGenerator gen("tf_test");

      for (int i = 0; i < 5; i++) {
        const auto insert_query = gen(i, i * 1.1, i * 1.1, 1.0 - i * 2.2);
        run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      }
    }
    {
      run_ddl_statement("DROP TABLE IF EXISTS sd_test;");
      run_ddl_statement(
          "CREATE TABLE sd_test ("
          "   base TEXT ENCODING DICT(32),"
          "   derived TEXT,"
          "   SHARED DICTIONARY (derived) REFERENCES sd_test(base)"
          ");");

      TestHelpers::ValuesGenerator gen("sd_test");
      std::vector<std::pair<std::string, std::string>> v = {{"'hello'", "'world'"},
                                                            {"'foo'", "'bar'"},
                                                            {"'bar'", "'baz'"},
                                                            {"'world'", "'foo'"},
                                                            {"'baz'", "'hello'"}};

      for (const auto& p : v) {
        const auto insert_query = gen(p.first, p.second);
        run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      }
    }
  }

  void TearDown() override {
    run_ddl_statement("DROP TABLE IF EXISTS tf_test;");
    run_ddl_statement("DROP TABLE IF EXISTS sd_test;");
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
    // Numeric constant to integer
    {
      const auto rows = run_multiple_agg(
          "SELECT out0 FROM TABLE(ct_binding_scalar_multiply(CURSOR(SELECT x FROM "
          "tf_test), 2.2));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));
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
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(42 * i));
      }
    }

    {
      const auto rows = run_multiple_agg(
          "SELECT answer / 882 AS g, COUNT(*) AS n FROM "
          "TABLE(ct_no_arg_constant_sizing()) GROUP BY g ORDER BY g;",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(2));

      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(0));
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]), static_cast<int32_t>(21));

      crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(1));
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]), static_cast<int32_t>(21));
    }

    {
      const auto rows =
          run_multiple_agg("SELECT answer FROM TABLE(ct_no_arg_runtime_sizing());", dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(42));
    }

    {
      const auto rows = run_multiple_agg(
          "SELECT answer FROM TABLE(ct_scalar_1_arg_runtime_sizing(123));", dt);
      ASSERT_EQ(rows->rowCount(), size_t(3));

      auto crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(123));
      crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(12));
      crt_row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(1));
    }

    {
      const auto rows = run_multiple_agg(
          "SELECT answer1, answer2 FROM TABLE(ct_scalar_2_args_constant_sizing(100, 5));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(5));

      for (size_t r = 0; r < 5; ++r) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int32_t>(100 + r * 5));
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]), static_cast<int32_t>(100 - r * 5));
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
  QR::init(BASE_PATH);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
