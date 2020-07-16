/*
 * Copyright 2020 OmniSci, Inc.
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
 * @file ForeignTableDmlTest.cpp
 * @brief Test suite for DML SQL queries on foreign tables
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "DBHandlerTestHelpers.h"
#include "DataMgr/ForeignStorage/ForeignStorageCache.h"
#include "DataMgr/ForeignStorage/ForeignStorageMgr.h"
#include "Shared/geo_types.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;
std::string test_binary_file_path;

namespace bp = boost::process;
namespace bf = boost::filesystem;
using path = bf::path;

static const std::string default_table_name = "test_foreign_table";

/**
 * Helper class for creating foreign tables
 */
class ForeignTableTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override { DBHandlerTestFixture::SetUp(); }
  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  static std::string getCreateForeignTableQuery(const std::string& columns,
                                                const std::string& file_name_base,
                                                const std::string& data_wrapper_type,
                                                const int table_number = 0) {
    return getCreateForeignTableQuery(
        columns, {}, file_name_base, data_wrapper_type, table_number);
  }

  static std::string getCreateForeignTableQuery(
      const std::string& columns,
      const std::map<std::string, std::string> options,
      const std::string& file_name_base,
      const std::string& data_wrapper_type,
      const int table_number = 0,
      const std::string& table_name = default_table_name) {
    std::string query{"CREATE FOREIGN TABLE " + table_name};
    if (table_number) {
      query += "_" + std::to_string(table_number);
    }
    query += columns + " SERVER omnisci_local_" + data_wrapper_type +
             " WITH (file_path = '" + getDataFilesPath() + file_name_base + "." +
             data_wrapper_type + "'";
    for (auto& [key, value] : options) {
      query += ", " + key + " = '" + value + "'";
    }
    query += ");";
    return query;
  }

  static std::string getDataFilesPath() {
    return bf::canonical(test_binary_file_path + "/../../Tests/FsiDataFiles").string() +
           "/";
  }

  static void sqlCreateForeignTable(const std::string& columns,
                                    const std::string& file_name,
                                    const std::string& data_wrapper_type,
                                    const int table_number = 0,
                                    const std::string& table_name = default_table_name) {
    sqlDropForeignTable(table_name);
    auto query = getCreateForeignTableQuery(
        columns, {}, file_name, data_wrapper_type, table_number, table_name);
    sql(query);
  }

  static void sqlDropForeignTable(const std::string& table_name = default_table_name) {
    sql("DROP FOREIGN TABLE IF EXISTS " + table_name);
  }
};

class SelectQueryTest : public ForeignTableTest {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table_2;");
    sql("DROP SERVER IF EXISTS test_server;");
  }

  void TearDown() override {
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table_2;");
    sql("DROP SERVER IF EXISTS test_server;");
    DBHandlerTestFixture::TearDown();
  }
};

class DataWrapperSelectQueryTest : public SelectQueryTest,
                                   public ::testing::WithParamInterface<std::string> {};

class DataTypeFragmentSizeTest : public SelectQueryTest,
                                 public testing::WithParamInterface<int> {};

class DataTypeFragmentSizeAndDataWrapperTest
    : public SelectQueryTest,
      public testing::WithParamInterface<std::pair<int, std::string>> {};

class RowGroupAndFragmentSizeSelectQueryTest
    : public SelectQueryTest,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {};

TEST_F(SelectQueryTest, CustomServer) {
  sql("CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "s +
      "WITH (storage_type = 'LOCAL_FILE', base_path = '" + getDataFilesPath() + "');");
  sql("CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER[]) "
      "SERVER test_server WITH (file_path = 'example_1.csv');");
  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_F(SelectQueryTest, DefaultLocalCsvServer) {
  std::string query = "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER[]) "s +
                      "SERVER omnisci_local_csv WITH (file_path = '" +
                      getDataFilesPath() + "/example_1.csv');";
  sql(query);
  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_F(SelectQueryTest, DefaultLocalParquetServer) {
  std::string query =
      "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER, f DOUBLE) "s +
      "SERVER omnisci_local_parquet WITH (file_path = '" + getDataFilesPath() +
      "/example_2.parquet');";
  sql(query);
  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", i(1), 1.1},
                        {"aa", i(1), 1.1},
                        {"aa", i(2), 2.2},
                        {"aaa", i(1), 1.1},
                        {"aaa", i(2), 2.2},
                        {"aaa", i(3), 3.3}},
                       result);
}

namespace {
struct PrintToStringParamName {
  template <class ParamType>
  std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
    std::stringstream ss;
    ss << info.param;
    return ss.str();
  }

  std::string operator()(
      const ::testing::TestParamInfo<std::pair<int, std::string>>& info) const {
    std::stringstream ss;
    ss << "Fragment_size_" << info.param.first << "_Data_wrapper_" << info.param.second;
    return ss.str();
  }

  std::string operator()(
      const ::testing::TestParamInfo<std::pair<int64_t, int64_t>>& info) const {
    std::stringstream ss;
    ss << "Rowgroup_size_" << info.param.first << "_Fragment_size_" << info.param.second;
    return ss.str();
  }
};
}  // namespace

// Create table with multiple fragments with file buffers less than size of a
// fragment Includes both fixed and variable length data
TEST_F(SelectQueryTest, MultipleDataBlocksPerFragment) {
  const auto& query =
      getCreateForeignTableQuery("(i INTEGER,  txt TEXT, txt_2 TEXT ENCODING NONE)",
                                 {{"buffer_size", "25"}, {"fragment_size", "64"}},
                                 "0_255",
                                 "csv");
  sql(query);

  // Check that data is correct
  {
    std::vector<std::vector<TargetValue>> expected_result_set;
    for (int number = 0; number < 256; number++) {
      expected_result_set.push_back(
          {i(number), std::to_string(number), std::to_string(number)});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM test_foreign_table ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }

  // Check that WHERE statements filter numerical data correctly
  {
    std::vector<std::vector<TargetValue>> expected_result_set;
    for (int number = 128; number < 256; number++) {
      expected_result_set.push_back(
          {i(number), std::to_string(number), std::to_string(number)});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM test_foreign_table  WHERE i >= 128 ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }
  {
    std::vector<std::vector<TargetValue>> expected_result_set;
    for (int number = 0; number < 128; number++) {
      expected_result_set.push_back(
          {i(number), std::to_string(number), std::to_string(number)});
    }
    TQueryResult result;
    sql(result, "SELECT * FROM test_foreign_table  WHERE i < 128 ORDER BY i;");
    assertResultSetEqual(expected_result_set, result);
  }
}

TEST_F(SelectQueryTest, ParquetGeoTypesMalformed) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON)",
      "geo_types.malformed",
      "parquet");
  sql(query);

  queryAndAssertException("SELECT * FROM test_foreign_table;",
                          "Exception: Failure to import geo column 'l' in table "
                          "'test_foreign_table' for row group 0 and row 1.");
}

TEST_F(SelectQueryTest, ParquetGeoTypesNull) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON)",
      "geo_types.null",
      "parquet");
  sql(query);

  queryAndAssertException("SELECT * FROM test_foreign_table;",
                          "Exception: Failure to import geo column 'l' in table "
                          "'test_foreign_table' for row group 0 and row 1.");
}

TEST_F(SelectQueryTest, ParquetNullRowgroups) {
  const auto& query =
      getCreateForeignTableQuery("(a SMALLINT, b SMALLINT)", "null_columns", "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({{i(NULL_SMALLINT),i(1)},
                        {i(NULL_SMALLINT),i(2)},
                        {i(NULL_SMALLINT),i(NULL_SMALLINT)},
                        {i(NULL_SMALLINT),i(NULL_SMALLINT)}},
                       result);
  // clang-format on
}

INSTANTIATE_TEST_SUITE_P(DataWrapperParameterizedTests,
                         DataWrapperSelectQueryTest,
                         ::testing::Values("csv", "parquet"),
                         PrintToStringParamName());

TEST_P(DataWrapperSelectQueryTest, AggregateAndGroupBy) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);

  TQueryResult result;
  sql(result, "SELECT t, avg(i), sum(f) FROM test_foreign_table group by t;");
  // clang-format off
  assertResultSetEqual({{"a", 1.0, 1.1},
                        {"aa", 1.5, 3.3},
                        {"aaa", 2.0, 6.6}},
                       result);
  // clang-format on
}

// TODO: implement for parquet when kARRAY support implemented for parquet
TEST_F(SelectQueryTest, Join) {
  auto query = getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1", "csv");
  sql(query);

  query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER, d DOUBLE)", "example_2", "csv", 2);
  sql(query);

  TQueryResult result;
  sql(result,
      "SELECT t1.t, t1.i, t2.i, t2.d FROM test_foreign_table AS t1 JOIN "
      "test_foreign_table_2 AS t2 ON t1.t = t2.t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)}), i(1), 1.1},
                        {"aa", array({Null_i, i(2), i(2)}), i(1), 1.1},
                        {"aa", array({Null_i, i(2), i(2)}), i(2), 2.2},
                        {"aaa", array({i(3), Null_i, i(3)}), i(1), 1.1},
                        {"aaa", array({i(3), Null_i, i(3)}), i(2), 2.2},
                        {"aaa", array({i(3), Null_i, i(3)}), i(3), 3.3}},
                       result);
}

TEST_P(DataWrapperSelectQueryTest, Filter) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table WHERE i > 1;");
  // clang-format off
  assertResultSetEqual({{"aa", i(2), 2.2},
                        {"aaa", i(2), 2.2},
                        {"aaa", i(3), 3.3}},
                       result);
  // clang-format on
}

// TODO: implement for parquet when kARRAY support implemented for parquet
TEST_F(SelectQueryTest, Sort) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table ORDER BY t DESC;");
  assertResultSetEqual({{"aaa", array({i(3), Null_i, i(3)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"a", array({i(1), i(1), i(1)})}},
                       result);
}

TEST_P(DataWrapperSelectQueryTest, Update) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);
  queryAndAssertException("UPDATE test_foreign_table SET t = 'abc';",
                          "Exception: DELETE, INSERT, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_P(DataWrapperSelectQueryTest, Insert) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);
  queryAndAssertException("INSERT INTO test_foreign_table VALUES('abc', null, null);",
                          "Exception: DELETE, INSERT, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_P(DataWrapperSelectQueryTest, InsertIntoSelect) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);
  queryAndAssertException(
      "INSERT INTO test_foreign_table SELECT * FROM test_foreign_table;",
      "Exception: DELETE, INSERT, OR UPDATE commands are not supported for "
      "foreign "
      "tables.");
}

TEST_P(DataWrapperSelectQueryTest, Delete) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, f DOUBLE)", "example_2", GetParam());
  sql(query);
  queryAndAssertException("DELETE FROM test_foreign_table WHERE t = 'a';",
                          "Exception: DELETE, INSERT, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_F(SelectQueryTest, CSV_CustomDelimiters) {
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN, i INTEGER, f FLOAT, t TIME, tp TIMESTAMP, d DATE, "
      "txt TEXT, txt_2 TEXT, i_arr INTEGER[], txt_arr TEXT[])",
      {{"delimiter", "|"}, {"array_delimiter", "_"}},
      "custom_delimiters",
      "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({
    {
      True, i(30000), 10.1f, "00:00:10", "1/1/2000 00:00:59", "1/1/2000", "text_1",
      "quoted text", array({i(1)}), array({"quoted text"})
    },
    {
      False, i(30500), 100.12f, "00:10:00", "6/15/2020 00:59:59", "6/15/2020", "text_2",
      "quoted text 2", array({i(1), i(2), i(3)}), array({"quoted text 2", "quoted text 3"})
    },
    {
      True, i(31000), 1000.123f, "10:00:00", "12/31/2500 23:59:59", "12/31/2500", "text_3",
      "quoted text 3", array({i(10), i(20), i(30)}), array({"quoted_text_4", "quoted_text_5"})
    }},
    result);
  // clang-format on
}

TEST_F(SelectQueryTest, CSV_CustomLineDelimiters) {
  const auto& query = getCreateForeignTableQuery("(b BOOLEAN, i INTEGER, t TEXT)",
                                                 {{"line_delimiter", "*"}},
                                                 "custom_line_delimiter",
                                                 "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({{True, i(1000), "text_1"},
                        {False, i(2000), "text_2"},
                        {True, i(3000), "text_3"}},
                       result);
  // clang-format on
}

// TODO: implement for parquet when kARRAY support implemented for parquet
// Note: only need to test array_marker and array_delimiter
TEST_F(SelectQueryTest, CSV_CustomMarkers) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, t2 TEXT, i INTEGER[])",
      {{"array_marker", "[]"}, {"escape", "\\"}, {"nulls", "NIL"}, {"quote", "|"}},
      "custom_markers",
      "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{Null, "quoted |text|", array({i(1)})},
                        {"text_1", "quoted text", array({i(1), i(2)})},
                        {Null, "\"quoted\" \"text\"", array({i(3), i(4), i(5)})}},
                       result);
}

TEST_F(SelectQueryTest, CSV_NoHeader) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER[])", {{"header", "false"}}, "no_header", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({i(2), i(2), i(2)})},
                        {"aaa", array({i(3), i(3), i(3)})}},
                       result);
}

TEST_F(SelectQueryTest, CSV_QuotedHeader) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "quoted_headers", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({i(2), i(2), i(2)})},
                        {"aaa", array({i(3), i(3), i(3)})}},
                       result);
}

TEST_F(SelectQueryTest, CSV_NonQuotedFields) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, t2 TEXT)", {{"quoted", "false"}}, "non_quoted", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({{"text_1", "\"text_1\""},
                        {"text_2", "\"text_2\""},
                        {"text_3", "\"text_3\""}},
                       result);
  // clang-format on
}

TEST_F(SelectQueryTest, WithBufferSizeOption) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER[])", {{"buffer_size", "25"}}, "example_1", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table ORDER BY t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_F(SelectQueryTest, ReverseLongitudeAndLatitude) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT)", {{"lonlat", "false"}}, "reversed_long_lat", "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({{"POINT (1 0)"},
                        {"POINT (2 1)"},
                        {"POINT (3 2)"}},
                       result);
  // clang-format on
}

class RefreshForeignTableTest : public ForeignTableTest {
 protected:
  std::string table_1_filename = "example_1";
  std::string table_2_filename = "example_1";

  void SetUp() override { ForeignTableTest::SetUp(); }

  void TearDown() override {
    sql("DROP FOREIGN TABLE IF EXISTS " + default_table_name + ";");
    sql("DROP FOREIGN TABLE IF EXISTS " + default_table_name + "_1;");
    ForeignTableTest::TearDown();
  }
};

// Refresh is not enabled yet, so currently we expect throws.
TEST_F(RefreshForeignTableTest, RefreshSingleTable) {
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_1_filename, "csv");
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_2_filename, "csv", 1);
  queryAndAssertException("REFRESH FOREIGN TABLES test_foreign_table",
                          "Exception: REFRESH FOREIGN TABLES is not yet implemented");
}

TEST_F(RefreshForeignTableTest, RefreshMultipleTables) {
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_1_filename, "csv");
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_2_filename, "csv", 1);
  queryAndAssertException(
      "REFRESH FOREIGN TABLES test_foreign_table, test_foreign_table_2",
      "Exception: REFRESH FOREIGN TABLES is not yet implemented");
}

TEST_F(RefreshForeignTableTest, RefreshEvictFalseCaps) {
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_1_filename, "csv");
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_2_filename, "csv", 1);
  queryAndAssertException(
      "REFRESH FOREIGN TABLES test_foreign_table WITH (EVICT='false')",
      "Exception: REFRESH FOREIGN TABLES is not yet implemented");
}

TEST_F(RefreshForeignTableTest, RefreshEvictFalse) {
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_1_filename, "csv");
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_2_filename, "csv", 1);
  queryAndAssertException(
      "REFRESH FOREIGN TABLES test_foreign_table WITH (evict='false')",
      "Exception: REFRESH FOREIGN TABLES is not yet implemented");
}

TEST_F(RefreshForeignTableTest, RefreshEvictTrue) {
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_1_filename, "csv");
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_2_filename, "csv", 1);
  queryAndAssertException("REFRESH FOREIGN TABLES test_foreign_table WITH (EVICT='true')",
                          "Exception: REFRESH FOREIGN TABLES is not yet implemented");
}

TEST_F(RefreshForeignTableTest, RefreshEvictTrueCaps) {
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_1_filename, "csv");
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", table_2_filename, "csv", 1);
  queryAndAssertException("REFRESH FOREIGN TABLES test_foreign_table WITH (EVICT='TRUE')",
                          "Exception: REFRESH FOREIGN TABLES is not yet implemented");
}

INSTANTIATE_TEST_SUITE_P(FragmentSize_Small_Default,
                         DataTypeFragmentSizeTest,
                         ::testing::Values(1, 2, 32000000));

INSTANTIATE_TEST_SUITE_P(DataTypeFragmentSizeAndDataWrapperParameterizedTests,
                         DataTypeFragmentSizeAndDataWrapperTest,
                         ::testing::Values(std::make_pair(1, "csv"),
                                           std::make_pair(1, "parquet"),
                                           std::make_pair(2, "csv"),
                                           std::make_pair(2, "parquet"),
                                           std::make_pair(32000000, "csv"),
                                           std::make_pair(32000000, "parquet")),
                         PrintToStringParamName());

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, ScalarTypes) {
  auto& param = GetParam();
  int fragment_size = param.first;
  std::string data_wrapper_type = param.second;
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
      "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
      "txt_2 TEXT ENCODING NONE)",
      {{"fragment_size", std::to_string(fragment_size)}},
      "scalar_types",
      data_wrapper_type);
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({
    {
      True, i(100), i(30000), i(2000000000), i(9000000000000000000), 10.1f, 100.1234, "00:00:10",
      "1/1/2000 00:00:59", "1/1/2000", "text_1", "quoted text"
    },
    {
      False, i(110), i(30500), i(2000500000), i(9000000050000000000), 100.12f, 2.1234, "00:10:00",
    "6/15/2020 00:59:59", "6/15/2020", "text_2", "quoted text 2"
    },
    {
      True, i(120), i(31000), i(2100000000), i(9100000000000000000), 1000.123f, 100.1, "10:00:00",
      "12/31/2500 23:59:59", "12/31/2500", "text_3", "quoted text 3"
    }},
    result);
  // clang-format on
}

// TODO: implement for parquet when kARRAY support implemented for parquet
TEST_P(DataTypeFragmentSizeTest, ArrayTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN[], t TINYINT[], s SMALLINT[], i INTEGER[], bi BIGINT[], f "
      "FLOAT[], "
      "tm "
      "TIME[], tp TIMESTAMP[], "
      "d DATE[], txt TEXT[], txt_2 TEXT[])",
      {{"fragment_size", std::to_string(GetParam())}},
      "array_types",
      "csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({
    {
      array({True}), array({i(50), i(100)}), array({i(30000), i(20000)}), array({i(2000000000)}),
      array({i(9000000000000000000)}), array({10.1f, 11.1f}), array({"00:00:10"}),
      array({"1/1/2000 00:00:59", "1/1/2010 00:00:59"}), array({"1/1/2000", "2/2/2000"}),
      array({"text_1"}), array({"quoted text"})
    },
    {
      array({False, True}), array({i(110)}), array({i(30500)}), array({i(2000500000)}),
      array({i(9000000050000000000)}), array({100.12f}), array({"00:10:00", "00:20:00"}),
      array({"6/15/2020 00:59:59"}), array({"6/15/2020"}),
      array({"text_2", "text_3"}), array({"quoted text 2"})
    },
    {
      array({True}), array({i(120)}), array({i(31000)}), array({i(2100000000), i(200000000)}),
      array({i(9100000000000000000), i(9200000000000000000)}), array({1000.123f}), array({"10:00:00"}),
      array({"12/31/2500 23:59:59"}), array({"12/31/2500"}),
      array({"text_4"}), array({"quoted text 3", "quoted text 4"})
    }},
    result);
  // clang-format on
}

TEST_P(DataTypeFragmentSizeAndDataWrapperTest, GeoTypes) {
  auto& param = GetParam();
  int fragment_size = param.first;
  std::string data_wrapper_type = param.second;
  const auto& query = getCreateForeignTableQuery(
      "(p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON)",
      {{"fragment_size", std::to_string(fragment_size)}},
      "geo_types",
      data_wrapper_type);
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  // clang-format off
  assertResultSetEqual({
    {
      "POINT (0 0)", "LINESTRING (0 0,0 0)", "POLYGON ((0 0,1 0,0 1,1 1,0 0))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)))"
    },
    {
      "POINT (1 1)", "LINESTRING (1 1,2 2,3 3)", "POLYGON ((5 4,7 4,6 5,5 4))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"
    },
    {
      "POINT (2 2)", "LINESTRING (2 2,3 3)", "POLYGON ((1 1,3 1,2 3,1 1))",
      "MULTIPOLYGON (((0 0,3 0,0 3,0 0)),((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"
    }},
    result);
  // clang-format on
}

INSTANTIATE_TEST_SUITE_P(RowGroupAndFragmentSizeParameterizedTests,
                         RowGroupAndFragmentSizeSelectQueryTest,
                         ::testing::Values(std::make_pair(1, 1),
                                           std::make_pair(1, 2),
                                           std::make_pair(2, 1),
                                           std::make_pair(2, 2)),
                         PrintToStringParamName());

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, MetadataOnlyCount) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT COUNT(*) FROM test_foreign_table;");
  assertResultSetEqual({{i(6)}}, result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, MetadataOnlyFilter) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  {
    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM test_foreign_table WHERE a > 2;");
    assertResultSetEqual({{i(4)}}, result);
  }

  {
    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM test_foreign_table WHERE d < 0;");
    assertResultSetEqual({{i(2)}}, result);
  }
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, Join) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  auto query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);
  query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, d DOUBLE)", "example_2", "parquet", 2);
  sql(query);

  TQueryResult result;
  sql(result,
      "SELECT t1.a, t1.b, t1.c, t1.d, t2.i, t2.d FROM "
      "test_foreign_table AS t1 JOIN "
      "test_foreign_table_2 AS t2 ON t1.a = t2.i ORDER BY t1.a;");

  assertResultSetEqual({{i(1), i(3), i(6), 7.1, i(1), 1.1},
                        {i(1), i(3), i(6), 7.1, i(1), 1.1},
                        {i(1), i(3), i(6), 7.1, i(1), 1.1},
                        {i(2), i(4), i(7), 0.000591, i(2), 2.2},
                        {i(2), i(4), i(7), 0.000591, i(2), 2.2},
                        {i(3), i(5), i(8), 1.1, i(3), 3.3}},
                       result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, Select) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual({{i(1), i(3), i(6), 7.1},
                        {i(2), i(4), i(7), 0.000591},
                        {i(3), i(5), i(8), 1.1},
                        {i(4), i(6), i(9), 0.022123},
                        {i(5), i(7), i(10), -1.},
                        {i(6), i(8), i(1), -100.}},
                       result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, Filter) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "example_row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c INTEGER, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table WHERE d < 0 ;");
  assertResultSetEqual({{i(5), i(7), i(10), -1.}, {i(6), i(8), i(1), -100.}}, result);
}

TEST_P(RowGroupAndFragmentSizeSelectQueryTest, NoStatistics) {
  auto param = GetParam();
  int64_t row_group_size = param.first;
  int64_t fragment_size = param.second;
  std::stringstream filename_stream;
  filename_stream << "no_stats.row_group_size." << row_group_size;
  const auto& query =
      getCreateForeignTableQuery("(a INTEGER, b INTEGER, c TEXT, d DOUBLE)",
                                 {{"fragment_size", std::to_string(fragment_size)}},
                                 filename_stream.str(),
                                 "parquet");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table;");
  assertResultSetEqual(
      {
          {i(1), i(3), "5", 7.1},
          {i(2), i(4), "stuff", NULL_DOUBLE},
          {i(3), i(5), "8", 1.1},
          {i(4), i(6), "9", 2.2123e-2},
          {i(5), i(7), "10", -1.},
          {i(6), i(8), "1", -100.},
      },
      result);
}

using namespace foreign_storage;
class ForeignStorageCacheQueryTest : public ForeignTableTest {
 protected:
  inline static const std::string table_2_filename = "example_2";
  inline static const std::string col_name1 = "col1";
  inline static const std::string col_name2 = "col2";
  inline static const std::string col_name3 = "col3";
  inline static Catalog_Namespace::Catalog* cat;
  inline static ForeignStorageCache* cache;
  inline static File_Namespace::GlobalFileMgr* gfm;
  inline static const TableDescriptor* td;
  inline static const ColumnDescriptor *cd1, *cd2, *cd3;
  inline static ChunkKey query_chunk_key1, query_chunk_key2, query_chunk_key3,
      query_table_prefix;

  static void SetUpTestSuite() {
    DBHandlerTestFixture::SetUpTestSuite();
    cat = &getCatalog();
    cache = cat->getDataMgr().getForeignStorageMgr()->getForeignStorageCache();
    gfm = cat->getDataMgr().getGlobalFileMgr();
    sqlDropForeignTable();
  }

  static void TearDownTestSuite() { DBHandlerTestFixture::TearDownTestSuite(); }

  static void createTestTable() {
    sqlCreateForeignTable(
        "(" + col_name1 + " TEXT, " + col_name2 + " INTEGER, " + col_name3 + " DOUBLE)",
        table_2_filename,
        "csv");
    td = cat->getMetadataForTable(default_table_name);
    cd1 = cat->getMetadataForColumn(td->tableId, col_name1);
    cd2 = cat->getMetadataForColumn(td->tableId, col_name2);
    cd3 = cat->getMetadataForColumn(td->tableId, col_name3);
    query_chunk_key1 = {cat->getCurrentDB().dbId, td->tableId, cd1->columnId, 0};
    query_chunk_key2 = {cat->getCurrentDB().dbId, td->tableId, cd2->columnId, 0};
    query_chunk_key3 = {cat->getCurrentDB().dbId, td->tableId, cd3->columnId, 0};
    query_table_prefix = {cat->getCurrentDB().dbId, td->tableId};
  }

  static void sqlSelect(const std::string& columns = "*",
                        const std::string& table_name = "test_foreign_table") {
    sql("SELECT " + columns + " FROM " + table_name + ";");
  }

  void SetUp() override {
    ForeignTableTest::SetUp();
    cache->clear();
    createTestTable();
  }

  void TearDown() override {
    sqlDropForeignTable();
    ForeignTableTest::TearDown();
  }
};

TEST_F(ForeignStorageCacheQueryTest, SelectPopulateChunks) {
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_FALSE(gfm->isBufferOnDevice(query_chunk_key2));
  sqlSelect();
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_TRUE(gfm->isBufferOnDevice(query_chunk_key2));
  AbstractBuffer* buffer = cache->getCachedChunkIfExists(query_chunk_key2);
  ASSERT_NE(buffer, nullptr);
  int8_t array[4];
  buffer->read(array, 4);
  int32_t val = 1;
  ASSERT_EQ(std::memcmp(array, &val, 4), 0);
}

TEST_F(ForeignStorageCacheQueryTest, CreatePopulateMetadata) {
  sqlDropForeignTable();
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key1));
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key2));
  ASSERT_FALSE(cache->isMetadataCached(query_chunk_key3));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(query_chunk_key1));
  ASSERT_FALSE(cache->hasCachedMetadataForKeyPrefix(query_table_prefix));
  createTestTable();
  ASSERT_TRUE(cache->isMetadataCached(query_chunk_key1));
  ASSERT_TRUE(cache->isMetadataCached(query_chunk_key2));
  ASSERT_TRUE(cache->isMetadataCached(query_chunk_key3));
  ASSERT_TRUE(cache->hasCachedMetadataForKeyPrefix(query_chunk_key1));
  ASSERT_TRUE(cache->hasCachedMetadataForKeyPrefix(query_table_prefix));
}

TEST_F(ForeignStorageCacheQueryTest, CacheEvictAfterDrop) {
  sqlSelect();
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 3U);
  sqlDropForeignTable();
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 0U);
}

// LRU specific tests (default algorithm).
TEST_F(ForeignStorageCacheQueryTest, LRUEvictChunkByAlgOrderOne) {
  sqlSelect(col_name3);
  sqlSelect(col_name2);
  sqlSelect(col_name1);
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  size_t old_limit = cache->getLimit();
  cache->setLimit(2U);
  ASSERT_FALSE(cache->getCachedChunkIfExists(query_chunk_key3));
  ASSERT_TRUE(cache->getCachedChunkIfExists(query_chunk_key2));
  ASSERT_TRUE(cache->getCachedChunkIfExists(query_chunk_key1));
  cache->setLimit(1U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  cache->setLimit(0U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  cache->setLimit(old_limit);
}

// Verify that the LRU algorithm is working if we use a different order.
TEST_F(ForeignStorageCacheQueryTest, LRUEvictChunkByAlgOrderTwo) {
  sqlSelect(col_name1);
  sqlSelect(col_name2);
  sqlSelect(col_name3);
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  size_t old_limit = cache->getLimit();
  cache->setLimit(2U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  cache->setLimit(1U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_NE(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  cache->setLimit(0U);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key1), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key2), nullptr);
  ASSERT_EQ(cache->getCachedChunkIfExists(query_chunk_key3), nullptr);
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  cache->setLimit(old_limit);
}

TEST_F(ForeignStorageCacheQueryTest, WideLogicalColumns) {
  cache->clear();
  ASSERT_EQ(cache->getNumCachedChunks(), 0U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 0U);
  sqlDropForeignTable();
  sqlCreateForeignTable("(t TEXT, i INTEGER[])", "example_1", "csv");
  sqlSelect();
  // Metadata and chunk size differ because the INTEGER[] logical col expands into two
  // physical columns.
  ASSERT_EQ(cache->getNumCachedChunks(), 3U);
  ASSERT_EQ(cache->getNumCachedMetadata(), 2U);
  sqlDropForeignTable();
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  g_enable_fsi = true;

  // get dirname of test binary
  test_binary_file_path = bf::canonical(argv[0]).parent_path().string();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_fsi = false;
  return err;
}
