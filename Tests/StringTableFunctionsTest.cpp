/*
 * Copyright 2017 MapD Technologies, Inc.
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

inline auto sql(const std::string& sql_stmt) {
  return QueryRunner::QueryRunner::get()->runSQL(
      sql_stmt, ExecutorDeviceType::CPU, true, true);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, true, true);
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

    for (const auto p : v) {
      const auto insert_query = gen(p.first, p.second);
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    }
  }

  void TearDown() override { run_ddl_statement("DROP TABLE IF EXISTS sd_test;"); }
};

TEST_F(TableFunctions, BasicProjection) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
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
      const auto rows = sql("SELECT base FROM sd_test;");
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
  }
}

TEST_F(TableFunctions, GroupByInAndOut) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
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
