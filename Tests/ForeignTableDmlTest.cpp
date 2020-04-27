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
#include "Shared/geo_types.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class SelectQueryTest : public DBHandlerTestFixture {
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

  std::string getCreateForeignTableQuery(const std::string& columns,
                                         const std::string& file_name,
                                         const int table_number = 0) {
    return getCreateForeignTableQuery(columns, {}, file_name, table_number);
  }

  std::string getCreateForeignTableQuery(const std::string& columns,
                                         const std::map<std::string, std::string> options,
                                         const std::string& file_name,
                                         const int table_number = 0) {
    boost::filesystem::path full_path =
        boost::filesystem::canonical("../../Tests/FsiDataFiles/" + file_name);
    std::string query{"CREATE FOREIGN TABLE test_foreign_table"};
    if (table_number) {
      query += "_" + std::to_string(table_number);
    }
    query += columns + " SERVER omnisci_local_csv WITH (file_path = '" +
             full_path.string() + "'";
    for (auto& [key, value] : options) {
      query += ", " + key + " = '" + value + "'";
    }
    query += ");";
    return query;
  }

  std::string getDataFilesPath() {
    return boost::filesystem::canonical("../../Tests/FsiDataFiles").string();
  }
};

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
      "CREATE FOREIGN TABLE test_foreign_table (t TEXT, i INTEGER, d DOUBLE) "
      "SERVER omnisci_local_parquet WITH (file_path = 'test_path');";
  sql(query);
  queryAndAssertException("SELECT * FROM test_foreign_table;",
                          "Exception: Unsupported data wrapper");
}

TEST_F(SelectQueryTest, ScalarTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
      "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
      "txt_2 TEXT ENCODING NONE)",
      "scalar_types.csv");
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

TEST_F(SelectQueryTest, ArrayTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(b BOOLEAN[], t TINYINT[], s SMALLINT[], i INTEGER[], bi BIGINT[], f FLOAT[], tm "
      "TIME[], tp TIMESTAMP[], "
      "d DATE[], txt TEXT[], txt_2 TEXT[])",
      "array_types.csv");
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

TEST_F(SelectQueryTest, GeoTypes) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON)", "geo_types.csv");
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

TEST_F(SelectQueryTest, AggregateAndGroupBy) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER, f DOUBLE)", "example_2.csv");
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

TEST_F(SelectQueryTest, Join) {
  auto query = getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1.csv");
  sql(query);

  query = getCreateForeignTableQuery("(t TEXT, i INTEGER, d DOUBLE)", "example_2.csv", 2);
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

TEST_F(SelectQueryTest, Filter) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER, f DOUBLE)", "example_2.csv");
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

TEST_F(SelectQueryTest, Sort) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1.csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table ORDER BY t DESC;");
  assertResultSetEqual({{"aaa", array({i(3), Null_i, i(3)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"a", array({i(1), i(1), i(1)})}},
                       result);
}

TEST_F(SelectQueryTest, Update) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1.csv");
  sql(query);
  queryAndAssertException("UPDATE test_foreign_table SET t = 'abc';",
                          "Exception: DELETE, INSERT, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_F(SelectQueryTest, Insert) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1.csv");
  sql(query);
  queryAndAssertException("INSERT INTO test_foreign_table VALUES('abc', null);",
                          "Exception: DELETE, INSERT, OR UPDATE commands are not "
                          "supported for foreign tables.");
}

TEST_F(SelectQueryTest, InsertIntoSelect) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1.csv");
  sql(query);
  queryAndAssertException(
      "INSERT INTO test_foreign_table SELECT * FROM test_foreign_table;",
      "Exception: DELETE, INSERT, OR UPDATE commands are not supported for foreign "
      "tables.");
}

TEST_F(SelectQueryTest, Delete) {
  const auto& query =
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "example_1.csv");
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
      "custom_delimiters.csv");
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
                                                 "custom_line_delimiter.csv");
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

TEST_F(SelectQueryTest, CSV_CustomMarkers) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, t2 TEXT, i INTEGER[])",
      {{"array_marker", "[]"}, {"escape", "\\"}, {"nulls", "NIL"}, {"quote", "|"}},
      "custom_markers.csv");
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
      "(t TEXT, i INTEGER[])", {{"header", "false"}}, "no_header.csv");
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
      getCreateForeignTableQuery("(t TEXT, i INTEGER[])", "quoted_headers.csv");
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
      "(t TEXT, t2 TEXT)", {{"quoted", "false"}}, "non_quoted.csv");
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
      "(t TEXT, i INTEGER[])", {{"buffer_size", "25"}}, "example_1.csv");
  sql(query);

  TQueryResult result;
  sql(result, "SELECT * FROM test_foreign_table ORDER BY t;");
  assertResultSetEqual({{"a", array({i(1), i(1), i(1)})},
                        {"aa", array({Null_i, i(2), i(2)})},
                        {"aaa", array({i(3), Null_i, i(3)})}},
                       result);
}

TEST_F(SelectQueryTest, WithFragmentSizeOption) {
  const auto& query = getCreateForeignTableQuery(
      "(t TEXT, i INTEGER, d DOUBLE)", {{"fragment_size", "2"}}, "example_2.csv");
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

TEST_F(SelectQueryTest, ReverseLongitudeAndLatitude) {
  const auto& query = getCreateForeignTableQuery(
      "(p POINT)", {{"lonlat", "false"}}, "reversed_long_lat.csv");
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

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  g_enable_fsi = true;

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_fsi = false;
  return err;
}
