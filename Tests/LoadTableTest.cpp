/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/writer.h>
#include <gtest/gtest.h>
#include <set>
#include <boost/filesystem.hpp>

#include "Geospatial/ColumnNames.h"
#include "Shared/ArrowUtil.h"
#include "Tests/SqlTestFixture.h"
#include "Tests/TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class LoadTableTest : public SqlTestFixture {
 protected:
  void SetUp() override {
    SqlTestFixture::SetUp();
    sql("DROP TABLE IF EXISTS load_test");
    sql("DROP TABLE IF EXISTS geo_load_test");
    sql("CREATE TABLE geo_load_test(i1 INTEGER, ls LINESTRING DEFAULT 'LINESTRING(0 0, 1 "
        "1)', s TEXT, mp MULTIPOLYGON, "
        "nns TEXT not null)");
    sql("CREATE TABLE load_test(i1 INTEGER, s TEXT DEFAULT 'default str' ENCODING "
        "DICT(8), nns TEXT not null)");
    initData();
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS load_test");
    sql("DROP TABLE IF EXISTS geo_load_test");
    SqlTestFixture::TearDown();
  }

  TStringValue getNullSV() const {
    TStringValue v;
    v.is_null = true;
    v.str_val = "";
    return v;
  }

  TStringValue getSV(const std::string& value) const {
    TStringValue v;
    v.is_null = false;
    v.str_val = value;
    return v;
  }

  const std::string LINESTRING = "LINESTRING (0 0,1 1,1 2)";
  const std::string DEFAULT_LINESTRING = "LINESTRING (0 0,1 1)";
  const std::string MULTIPOLYGON =
      "MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0),"
      "(1 1,1 2,2 2,2 1,1 1)),((-1 -1,-2 -1,-2 -2,-1 -2,-1 -1)))";
  const std::string POLYGON = "POLYGON ((0 0,4 0,4 4,0 4,0 0),(1 1,1 2,2 2,2 1,1 1))";
  const std::string POLYGON_PROMOTED =
      "MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0),(1 1,1 2,2 2,2 1,1 1)))";

  TColumn i1_column, s_column, nns_column, ls_column, mp_column, p_column;
  TDatum i1_datum, s_datum, nns_datum, ls_datum, mp_datum, p_datum;
  std::shared_ptr<arrow::Field> i1_field, s_field, nns_field, ls_field, mp_field, p_field;

 private:
  void initData() {
    p_column.nulls = std::vector<bool>(1, false);
    i1_column.nulls = s_column.nulls = nns_column.nulls = ls_column.nulls =
        mp_column.nulls = p_column.nulls;
    i1_column.data.int_col = {1};
    s_column.data.str_col = {"s"};
    nns_column.data.str_col = {"nns"};
    ls_column.data.str_col = {LINESTRING};
    mp_column.data.str_col = {MULTIPOLYGON};
    p_column.data.str_col = {POLYGON};

    i1_datum.is_null = s_datum.is_null = nns_datum.is_null = ls_datum.is_null =
        mp_datum.is_null = p_datum.is_null = false;
    i1_datum.val.int_val = 1;
    s_datum.val.str_val = "s";
    nns_datum.val.str_val = "nns";
    ls_datum.val.str_val = LINESTRING;
    mp_datum.val.str_val = MULTIPOLYGON;
    p_datum.val.str_val = POLYGON;

    i1_field = arrow::field("i1", arrow::int32());
    s_field = arrow::field("s", arrow::utf8());
    nns_field = arrow::field("nns", arrow::utf8());
    ls_field = arrow::field("ls", arrow::utf8());
    mp_field = arrow::field("mp", arrow::utf8());
    p_field = arrow::field("mp", arrow::utf8());
  }
};

TEST_F(LoadTableTest, AllColumnsNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TStringRow row;
  row.cols = {getSV("1"), getSV("s"), getSV("nns")};
  handler->load_table(session, "load_test", {row}, {});
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "s", "nns"}});
}

TEST_F(LoadTableTest, AllColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TStringRow row;
  row.cols = {
      getSV("1"), getSV(LINESTRING), getSV("s"), getSV(MULTIPOLYGON), getSV("nns")};
  handler->load_table(session, "geo_load_test", {row}, {});
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", MULTIPOLYGON, "nns"}});
}

TEST_F(LoadTableTest, AllColumnsPromotePolyToMPoly) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TStringRow row;
  row.cols = {getSV("1"), getSV(LINESTRING), getSV("s"), getSV(POLYGON), getSV("nns")};
  handler->load_table(session, "geo_load_test", {row}, {});
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", POLYGON_PROMOTED, "nns"}});
}

TEST_F(LoadTableTest, AllColumnsReordered) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "nns", "ls", "i1", "s"};
  TStringRow row;
  row.cols = {
      getSV(MULTIPOLYGON), getSV("nns"), getSV(LINESTRING), getSV("1"), getSV("s")};
  handler->load_table(session, "geo_load_test", {row}, column_names);
  sqlAndCompareResult("SELECT mp, nns, ls, i1, s FROM geo_load_test",
                      {{MULTIPOLYGON, "nns", LINESTRING, i(1), "s"}});
}

TEST_F(LoadTableTest, SomeColumnsReordered) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "nns", "ls"};
  TStringRow row;
  row.cols = {getSV(MULTIPOLYGON), getSV("nns"), getSV(LINESTRING)};
  handler->load_table(session, "geo_load_test", {row}, column_names);
  sqlAndCompareResult("SELECT mp, nns, ls, i1, s FROM geo_load_test",
                      {{MULTIPOLYGON, "nns", LINESTRING, nullptr, nullptr}});
}

TEST_F(LoadTableTest, OmitNotNullableColumn) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "ls"};
  TStringRow row;
  row.cols = {getSV(MULTIPOLYGON), getSV(LINESTRING)};
  executeLambdaAndAssertException(
      [&]() { handler->load_table(session, "geo_load_test", {row}, column_names); },
      "TException - service has thrown: TDBException(error_msg="
      "Column 'nns' cannot be omitted due to NOT NULL constraint)");
}

TEST_F(LoadTableTest, OmitGeoColumn) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"i1", "s", "nns", "ls"};
  TStringRow row;
  row.cols = {getSV("1"), getSV("s"), getSV("nns"), getSV(LINESTRING)};
  handler->load_table(session, "geo_load_test", {row}, column_names);
  sqlAndCompareResult("SELECT i1, s, nns, mp, ls FROM geo_load_test",
                      {{i(1), "s", "nns", (void*)0, LINESTRING}});
}

TEST_F(LoadTableTest, DuplicateColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "ls", "mp", "nns"};
  TStringRow row;
  row.cols = {getSV(MULTIPOLYGON), getSV(LINESTRING), getSV(MULTIPOLYGON), getSV("nns")};
  executeLambdaAndAssertException(
      [&]() { handler->load_table(session, "geo_load_test", {row}, column_names); },
      "TException - service has thrown: TDBException(error_msg="
      "Column mp is mentioned multiple times)");
}

TEST_F(LoadTableTest, UnexistingColumn) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "ls", "mp2", "nns"};
  TStringRow row;
  row.cols = {getSV(MULTIPOLYGON), getSV(LINESTRING), getSV(MULTIPOLYGON), getSV("nns")};
  executeLambdaAndAssertException(
      [&]() { handler->load_table(session, "geo_load_test", {row}, column_names); },
      "TException - service has thrown: TDBException(error_msg="
      "Column mp2 does not exist)");
}

TEST_F(LoadTableTest, ColumnNumberMismatch) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "ls", "i1", "nns"};
  TStringRow row;
  row.cols = {getSV(MULTIPOLYGON), getSV(LINESTRING), getSV("nns")};
  executeLambdaAndAssertException(
      [&]() { handler->load_table(session, "geo_load_test", {row}, column_names); },
      "TException - service has thrown: TDBException(error_msg="
      "Number of columns specified does not match the number of columns given (3 vs 4))");
}

TEST_F(LoadTableTest, NoColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  executeLambdaAndAssertException(
      [&]() { handler->load_table(session, "geo_load_test", {}, {}); },
      "TException - service has thrown: TDBException(error_msg="
      "No rows to insert)");
}

TEST_F(LoadTableTest, DefaultString) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"i1", "nns"};
  TStringRow row;
  row.cols = {getSV("1"), getSV("nns")};
  handler->load_table(session, "load_test", {row}, column_names);
  sqlAndCompareResult("SELECT i1, s, nns FROM load_test", {{i(1), "default str", "nns"}});
}

TEST_F(LoadTableTest, DefaultGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"i1", "s", "nns", "mp"};
  TStringRow row;
  row.cols = {getSV("1"), getSV("s"), getSV("nns"), getSV(MULTIPOLYGON)};
  handler->load_table(session, "geo_load_test", {row}, column_names);
  sqlAndCompareResult("SELECT i1, s, nns, mp, ls FROM geo_load_test",
                      {{i(1), "s", "nns", MULTIPOLYGON, DEFAULT_LINESTRING}});
}

TEST_F(LoadTableTest, BinaryAllColumnsNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRow row;
  row.cols = {i1_datum, s_datum, nns_datum};
  handler->load_table_binary(session, "load_test", {row}, {});
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "s", "nns"}});
}

TEST_F(LoadTableTest, DictOutOfBounds) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<TRow> rows;
  for (int i = 0; i < 300; i++) {
    TRow row;
    TDatum str_datum;
    str_datum.is_null = false;
    i1_datum.val.int_val = 1;
    str_datum.val.str_val = std::to_string(i);

    row.cols = {i1_datum, str_datum, nns_datum};
    rows.emplace_back(row);
  }
  executeLambdaAndAssertPartialException(
      [&]() { handler->load_table_binary(session, "load_test", rows, {}); },
      "has exceeded it's limit of 8 bits (255 unique values) while attempting to "
      "add the new string '255'. To load more data, please re-create the table "
      "with this column as type TEXT ENCODING DICT(16) or TEXT ENCODING DICT(32) "
      "and reload your data.");

  sqlAndCompareResult("SELECT count(*) FROM load_test", {{i(0)}});
}

// TODO(max): load_table_binary doesn't support tables with geo columns yet
TEST_F(LoadTableTest, DISABLED_BinaryAllColumnsPromotePolyToMPoly) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRow row;
  row.cols = {i1_datum, ls_datum, s_datum, p_datum, nns_datum};
  handler->load_table_binary(session, "geo_load_test", {row}, {});
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", POLYGON_PROMOTED, "nns"}});
}

TEST_F(LoadTableTest, DISABLED_BinaryAllColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRow row;
  row.cols = {i1_datum, ls_datum, s_datum, mp_datum, nns_datum};
  handler->load_table_binary(session, "geo_load_test", {row}, {});
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", MULTIPOLYGON, "nns"}});
}

TEST_F(LoadTableTest, BinaryAllColumnsReorderedNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"nns", "i1", "s"};
  TRow row;
  row.cols = {nns_datum, i1_datum, s_datum};
  handler->load_table_binary(session, "load_test", {row}, column_names);
  sqlAndCompareResult("SELECT i1, s, nns FROM load_test", {{i(1), "s", "nns"}});
}

TEST_F(LoadTableTest, BinarySomeColumnsReorderedNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"nns", "s"};
  TRow row;
  row.cols = {nns_datum, s_datum};
  handler->load_table_binary(session, "load_test", {row}, column_names);
  sqlAndCompareResult("SELECT i1, s, nns FROM load_test", {{nullptr, "s", "nns"}});
}

TEST_F(LoadTableTest, BinaryOmitNotNullableColumnNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"i1", "s"};
  TRow row;
  row.cols = {i1_datum, s_datum};
  executeLambdaAndAssertException(
      [&]() { handler->load_table_binary(session, "load_test", {row}, column_names); },
      "TException - service has thrown: TDBException(error_msg="
      "Column 'nns' cannot be omitted due to NOT NULL constraint)");
}

TEST_F(LoadTableTest, BinaryDuplicateColumnsNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"nns", "i1", "i1"};
  TRow row;
  row.cols = {nns_datum, i1_datum, i1_datum};
  executeLambdaAndAssertException(
      [&]() { handler->load_table_binary(session, "load_test", {row}, column_names); },
      "TException - service has thrown: TDBException(error_msg="
      "Column i1 is mentioned multiple times)");
}

TEST_F(LoadTableTest, BinaryUnexistingColumnNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"nns", "i1", "i2"};
  TRow row;
  row.cols = {nns_datum, i1_datum, i1_datum};
  executeLambdaAndAssertException(
      [&]() { handler->load_table_binary(session, "load_test", {row}, column_names); },
      "TException - service has thrown: TDBException(error_msg="
      "Column i2 does not exist)");
}

TEST_F(LoadTableTest, BinaryColumnNumberMismatchNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"nns", "i1", "s"};
  TRow row;
  row.cols = {nns_datum, i1_datum};
  executeLambdaAndAssertException(
      [&]() { handler->load_table_binary(session, "load_test", {row}, column_names); },
      "TException - service has thrown: TDBException(error_msg="
      "Number of columns specified does not match the number of columns given "
      "(2 vs 3))");
}

TEST_F(LoadTableTest, BinaryNoColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  executeLambdaAndAssertException(
      [&]() { handler->load_table_binary(session, "load_test", {}, {}); },
      "TException - service has thrown: TDBException(error_msg="
      "No rows to insert)");
}

TEST_F(LoadTableTest, BinaryDefaultString) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"i1", "nns"};
  TRow row;
  row.cols = {i1_datum, nns_datum};
  handler->load_table_binary(session, "load_test", {row}, column_names);
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "default str", "nns"}});
}

TEST_F(LoadTableTest, ColumnarNullsEmpty) {
  executeLambdaAndAssertException(
      [&]() {
        std::vector<std::string> column_names{"nns"};
        auto [handler, session] = getDbHandlerAndSessionId();
        auto nns_nulls_empty = nns_column;
        nns_nulls_empty.nulls.clear();
        handler->load_table_binary_columnar(
            session, "load_test", {nns_nulls_empty}, column_names);
      },
      "load_table_binary_columnar: Input exception thrown: Column nulls vector and data "
      "vector sizes do not match.. Issue at column : 3. Import aborted");
  sqlAndCompareResult("SELECT * FROM load_test;", {});
}

TEST_F(LoadTableTest, ColumnarNullsCountMismatch) {
  executeLambdaAndAssertException(
      [&]() {
        std::vector<std::string> column_names{"nns"};
        auto [handler, session] = getDbHandlerAndSessionId();
        auto nns = nns_column;
        nns.data.str_col.emplace_back("nns");
        // Do not add null.
        handler->load_table_binary_columnar(session, "load_test", {nns}, column_names);
      },
      "load_table_binary_columnar: Input exception thrown: Column nulls vector and data "
      "vector sizes do not match.. Issue at column : 3. Import aborted");
  sqlAndCompareResult("SELECT * FROM load_test;", {});
}

TEST_F(LoadTableTest, ColumnarInconsistentRowCount) {
  executeLambdaAndAssertPartialException(
      [&]() {
        std::vector<std::string> column_names{"nns", "i1"};
        auto [handler, session] = getDbHandlerAndSessionId();
        auto nns = nns_column;
        nns.data.str_col.emplace_back("nns");
        nns.nulls.emplace_back(false);
        handler->load_table_binary_columnar(
            session, "load_test", {nns, i1_column}, column_names);
      },
      "load_table_binary_columnar: Inconsistent number of rows in column i1 ,  expecting "
      "2 rows, column 0 has 1 rows). Issue at column : 1. Import aborted");
  sqlAndCompareResult("SELECT * FROM load_test;", {});
}

TEST_F(LoadTableTest, ColumnarWrongColumnTypeData) {
  executeLambdaAndAssertException(
      [&]() {
        std::vector<std::string> column_names{"nns"};
        auto [handler, session] = getDbHandlerAndSessionId();
        auto nns = nns_column;
        // Add an integer row instead of a string row.
        nns.data.int_col.emplace_back(10);
        nns.nulls.emplace_back(false);
        handler->load_table_binary_columnar(session, "load_test", {nns}, column_names);
      },
      "load_table_binary_columnar: Input exception thrown: Column data set for the wrong "
      "type.. Issue at column : 3. Import aborted");
  sqlAndCompareResult("SELECT * FROM load_test;", {});
}

TEST_F(LoadTableTest, ColumnarCorrectAndWrongColumnTypeData) {
  executeLambdaAndAssertException(
      [&]() {
        std::vector<std::string> column_names{"nns"};
        auto [handler, session] = getDbHandlerAndSessionId();
        auto nns = nns_column;
        // Correctly add a string row, but also incorrectly add an integer row.
        nns.data.str_col.emplace_back("nns");
        nns.data.int_col.emplace_back(10);
        nns.nulls.emplace_back(false);
        handler->load_table_binary_columnar(session, "load_test", {nns}, column_names);
      },
      "load_table_binary_columnar: Input exception thrown: Column data set for the wrong "
      "type.. Issue at column : 3. Import aborted");
  sqlAndCompareResult("SELECT * FROM load_test;", {});
}

TEST_F(LoadTableTest, ColumnarWrongColumnTypeDataAndMissingNullFlag) {
  executeLambdaAndAssertException(
      [&]() {
        std::vector<std::string> column_names{"nns"};
        auto [handler, session] = getDbHandlerAndSessionId();
        auto nns = nns_column;
        // Add an integer row instead of a string row.
        nns.data.int_col.emplace_back(10);
        handler->load_table_binary_columnar(session, "load_test", {nns}, column_names);
      },
      "load_table_binary_columnar: Input exception thrown: Column data set for the wrong "
      "type.. Issue at column : 3. Import aborted");
  sqlAndCompareResult("SELECT * FROM load_test;", {});
}

TEST_F(LoadTableTest, ColumnarAllColumnsNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  handler->load_table_binary_columnar(
      session, "load_test", {i1_column, s_column, nns_column}, {});
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "s", "nns"}});
}

TEST_F(LoadTableTest, ColumnarAllColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  handler->load_table_binary_columnar(
      session,
      "geo_load_test",
      {i1_column, ls_column, s_column, mp_column, nns_column},
      {});
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", MULTIPOLYGON, "nns"}});
}

TEST_F(LoadTableTest, ColumnarAllColumnsPromotePolyToMPoly) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  handler->load_table_binary_columnar(
      session,
      "geo_load_test",
      {i1_column, ls_column, s_column, p_column, nns_column},
      {});
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", POLYGON_PROMOTED, "nns"}});
}

TEST_F(LoadTableTest, ColumnarAllColumnsReordered) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "nns", "ls", "i1", "s"};
  handler->load_table_binary_columnar(
      session,
      "geo_load_test",
      {mp_column, nns_column, ls_column, i1_column, s_column},
      column_names);
  sqlAndCompareResult("SELECT mp, nns, ls, i1, s FROM geo_load_test",
                      {{MULTIPOLYGON, "nns", LINESTRING, i(1), "s"}});
}

TEST_F(LoadTableTest, ColumnarSomeColumnsReordered) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "nns", "ls"};
  handler->load_table_binary_columnar(
      session, "geo_load_test", {mp_column, nns_column, ls_column}, column_names);
  sqlAndCompareResult("SELECT mp, nns, ls, i1, s FROM geo_load_test",
                      {{MULTIPOLYGON, "nns", LINESTRING, nullptr, nullptr}});
}

TEST_F(LoadTableTest, ColumnarOmitNotNullableColumn) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "ls"};
  executeLambdaAndAssertException(
      [&]() {
        handler->load_table_binary_columnar(
            session, "geo_load_test", {mp_column, ls_column}, column_names);
      },
      "Column 'nns' cannot be omitted due to NOT NULL constraint");
}

TEST_F(LoadTableTest, ColumnarOmitGeoColumn) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"i1", "s", "nns", "ls"};
  handler->load_table_binary_columnar(session,
                                      "geo_load_test",
                                      {i1_column, s_column, nns_column, ls_column},
                                      column_names);
  sqlAndCompareResult("SELECT i1, s, nns, mp, ls FROM geo_load_test",
                      {{i(1), "s", "nns", (void*)0, LINESTRING}});
}

TEST_F(LoadTableTest, ColumnarDuplicateColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "ls", "mp"};
  executeLambdaAndAssertException(
      [&]() {
        handler->load_table_binary_columnar(
            session, "geo_load_test", {mp_column, ls_column, mp_column}, column_names);
      },
      "Column mp is mentioned multiple times");
}

TEST_F(LoadTableTest, ColumnarUnexistingColumn) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "ls", "mp2"};
  executeLambdaAndAssertException(
      [&]() {
        handler->load_table_binary_columnar(
            session, "geo_load_test", {mp_column, ls_column, mp_column}, column_names);
      },
      "Column mp2 does not exist");
}

TEST_F(LoadTableTest, ColumnarColumnNumberMismatch) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"mp", "ls", "i1"};
  executeLambdaAndAssertException(
      [&]() {
        handler->load_table_binary_columnar(
            session, "geo_load_test", {mp_column, ls_column}, column_names);
      },
      "Number of columns specified does not match the number of columns given (2 vs 3)");
}

TEST_F(LoadTableTest, ColumnarNoColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  executeLambdaAndAssertException(
      [&]() { handler->load_table_binary_columnar(session, "geo_load_test", {}, {}); },
      "No columns to insert");
}

TEST_F(LoadTableTest, ColumnarDefaultStr) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"i1", "nns"};
  handler->load_table_binary_columnar(
      session, "load_test", {i1_column, nns_column}, column_names);
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "default str", "nns"}});
}

TEST_F(LoadTableTest, ColumnarDefaultGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::vector<std::string> column_names{"i1", "s", "mp", "nns"};
  handler->load_table_binary_columnar(session,
                                      "geo_load_test",
                                      {i1_column, s_column, mp_column, nns_column},
                                      column_names);
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), DEFAULT_LINESTRING, "s", MULTIPOLYGON, "nns"}});
}

// A small helper to build Arrow stream for load_table_binary_arrow
class ArrowStreamBuilder {
 public:
  ArrowStreamBuilder(const std::shared_ptr<arrow::Schema>& schema) : schema_(schema) {}

  std::string finish() {
    CHECK(columns_.size() == schema_->fields().size());
    size_t length = columns_.empty() ? 0 : columns_[0]->length();
    auto records = arrow::RecordBatch::Make(schema_, length, columns_);
    auto out_stream = *arrow::io::BufferOutputStream::Create();
    auto stream_writer = *arrow::ipc::MakeStreamWriter(out_stream.get(), schema_);
    ARROW_THROW_NOT_OK(stream_writer->WriteRecordBatch(*records));
    ARROW_THROW_NOT_OK(stream_writer->Close());
    auto buffer = *out_stream->Finish();
    columns_.clear();
    return buffer->ToString();
  }

  void appendInt32(const std::vector<int32_t>& values,
                   const std::vector<bool>& is_null = {}) {
    append<arrow::Int32Builder, int32_t>(values, is_null);
  }
  void appendString(const std::vector<std::string>& values,
                    const std::vector<bool>& is_null = {}) {
    append<arrow::StringBuilder, std::string>(values, is_null);
  }

 private:
  template <typename Builder, typename T>
  void append(const std::vector<T>& values, const std::vector<bool>& is_null) {
    Builder builder;
    CHECK(is_null.empty() || values.size() == is_null.size());
    for (size_t i = 0; i < values.size(); ++i) {
      if (!is_null.empty() && is_null[i]) {
        ARROW_THROW_NOT_OK(builder.AppendNull());
      } else {
        ARROW_THROW_NOT_OK(builder.Append(values[i]));
      }
    }
    columns_.push_back(nullptr);
    ARROW_THROW_NOT_OK(builder.Finish(&columns_.back()));
  }

  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::Array>> columns_;
};

TEST_F(LoadTableTest, ArrowAllColumnsNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({i1_field, s_field, nns_field});
  ArrowStreamBuilder builder(schema);
  builder.appendInt32({1});
  builder.appendString({"s"});
  builder.appendString({"nns"});
  handler->load_table_binary_arrow(session, "load_test", builder.finish(), false);
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "s", "nns"}});
}

TEST_F(LoadTableTest, ArrowAllColumnsPromotePolyToMPoly) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({i1_field, ls_field, s_field, p_field, nns_field});
  ArrowStreamBuilder builder(schema);
  builder.appendInt32({1});
  builder.appendString({LINESTRING});
  builder.appendString({"s"});
  builder.appendString({POLYGON});
  builder.appendString({"nns"});
  handler->load_table_binary_arrow(session, "geo_load_test", builder.finish(), false);
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", POLYGON_PROMOTED, "nns"}});
}

TEST_F(LoadTableTest, ArrowAllColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({i1_field, ls_field, s_field, mp_field, nns_field});
  ArrowStreamBuilder builder(schema);
  builder.appendInt32({1});
  builder.appendString({LINESTRING});
  builder.appendString({"s"});
  builder.appendString({MULTIPOLYGON});
  builder.appendString({"nns"});
  handler->load_table_binary_arrow(session, "geo_load_test", builder.finish(), false);
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", MULTIPOLYGON, "nns"}});
}

TEST_F(LoadTableTest, ArrowAllColumnsReorderedNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({nns_field, i1_field, s_field});
  ArrowStreamBuilder builder(schema);
  builder.appendString({"nns"});
  builder.appendInt32({1});
  builder.appendString({"s"});
  handler->load_table_binary_arrow(session, "load_test", builder.finish(), true);
  sqlAndCompareResult("SELECT i1, s, nns FROM load_test", {{i(1), "s", "nns"}});
}

TEST_F(LoadTableTest, ArrowSomeColumnsReorderedNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({nns_field, s_field});
  ArrowStreamBuilder builder(schema);
  builder.appendString({"nns"});
  builder.appendString({"s"});
  handler->load_table_binary_arrow(session, "load_test", builder.finish(), true);
  sqlAndCompareResult("SELECT i1, s, nns FROM load_test", {{nullptr, "s", "nns"}});
}

TEST_F(LoadTableTest, ArrowOmitNotNullableColumnNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({i1_field, s_field});
  ArrowStreamBuilder builder(schema);
  builder.appendInt32({1});
  builder.appendString({"s"});
  executeLambdaAndAssertException(
      [&]() {
        handler->load_table_binary_arrow(session, "load_test", builder.finish(), true);
      },
      "Column 'nns' cannot be omitted due to NOT NULL constraint");
}

TEST_F(LoadTableTest, ArrowDuplicateColumnsNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({nns_field, i1_field, i1_field});
  ArrowStreamBuilder builder(schema);
  builder.appendString({"nns"});
  builder.appendInt32({1});
  builder.appendInt32({1});
  executeLambdaAndAssertException(
      [&]() {
        handler->load_table_binary_arrow(session, "load_test", builder.finish(), true);
      },
      "Column i1 is mentioned multiple times");
}

TEST_F(LoadTableTest, ArrowUnexistingColumnNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  std::shared_ptr<arrow::Field> i2_field = arrow::field("i2", arrow::int32());
  auto bad_schema = arrow::schema({nns_field, i1_field, i2_field});
  ArrowStreamBuilder builder(bad_schema);
  builder.appendString({"nns"});
  builder.appendInt32({1});
  builder.appendInt32({2});
  executeLambdaAndAssertException(
      [&]() {
        handler->load_table_binary_arrow(session, "load_test", builder.finish(), true);
      },
      "Column i2 does not exist");
}

TEST_F(LoadTableTest, ArrowNoColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({});
  ArrowStreamBuilder builder(schema);
  executeLambdaAndAssertException(
      [&]() {
        handler->load_table_binary_arrow(session, "load_test", builder.finish(), false);
      },
      "No columns to insert");
}

TEST_F(LoadTableTest, ArrowDefaultStr) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  auto schema = arrow::schema({i1_field, nns_field});
  ArrowStreamBuilder builder(schema);
  builder.appendInt32({1});
  builder.appendString({"nns"});
  handler->load_table_binary_arrow(session, "load_test", builder.finish(), true);
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "default str", "nns"}});
}

namespace {
using TablePrefix = std::pair<int64_t, int64_t>;  // {db_id, table_id}

// Visits the chunk key prefix and is_free flag of every buffer in the CPU (and GPU, if
// present) caches that belongs to a table.
template <typename Visitor>
void for_each_table_buffer(Visitor visitor) {
  std::vector<std::string> memory_levels{"cpu"};
  if (SqlTestFixture::gpusPresent()) {
    memory_levels.emplace_back("gpu");
  }
  auto [client, session] = SqlTestFixture::getDbHandlerAndSessionId();
  for (const auto& memory_level : memory_levels) {
    std::vector<TNodeMemoryInfo> memory_info;
    client->get_memory(memory_info, session, memory_level);
    for (const auto& node : memory_info) {
      for (const auto& memory_data : node.node_memory_data) {
        if (memory_data.chunk_key.size() >= 2) {
          visitor(TablePrefix{memory_data.chunk_key[0], memory_data.chunk_key[1]},
                  memory_data.is_free);
        }
      }
    }
  }
}

std::set<TablePrefix> get_table_prefixes() {
  std::set<TablePrefix> prefixes;
  for_each_table_buffer(
      [&prefixes](const TablePrefix& prefix, const bool) { prefixes.insert(prefix); });
  return prefixes;
}

std::vector<bool> get_buffer_free_flags(const TablePrefix& table_prefix) {
  std::vector<bool> flags;
  for_each_table_buffer([&](const TablePrefix& prefix, const bool is_free) {
    if (prefix == table_prefix) {
      flags.push_back(is_free);
    }
  });
  return flags;
}
}  // namespace

TEST_F(LoadTableTest, TemporaryTableChunkBufferCleanup) {
  // create a temporary table with multiple fragments to make that we're not leaking any
  // buffers on fragment creation
  sql("drop table if exists temp_table_cleanup_chuk_buffers;");
  const auto prefixes_before = get_table_prefixes();
  sql("create temporary table temp_table_cleanup_chuk_buffers (i1 integer) with "
      "(FRAGMENT_SIZE=1);");
  sql("insert into temp_table_cleanup_chuk_buffers values(1);");
  sql("insert into temp_table_cleanup_chuk_buffers values(2);");

  // Table ids are not exposed for temporary tables, so find the table as the only new
  // {db_id, table_id} prefix in the buffer pool.
  std::set<TablePrefix> new_prefixes;
  for (const auto& prefix : get_table_prefixes()) {
    if (!prefixes_before.count(prefix)) {
      new_prefixes.insert(prefix);
    }
  }
  ASSERT_EQ(new_prefixes.size(), size_t(1));
  const auto table_prefix = *new_prefixes.begin();

  // four buffers in use
  ASSERT_EQ(get_buffer_free_flags(table_prefix),
            (std::vector<bool>{false, false, false, false}));

  sql("drop table temp_table_cleanup_chuk_buffers;");
  // dropping the table leaves one freed buffer behind
  ASSERT_EQ(get_buffer_free_flags(table_prefix), (std::vector<bool>{true}));
}

class ImportGeoTableTest : public SqlTestFixture {
 protected:
  void SetUp() override {
    SqlTestFixture::SetUp();
    sql("DROP TABLE IF EXISTS import_geo_table_test");
  }

  const std::string getGeoFileName() const {
    auto geo_file_name =
        boost::filesystem::absolute(
            boost::filesystem::path(
                "../../Tests/ImportGeoTableTest/datafiles/geospatial_poly.geojson"))
            .string();
    return geo_file_name;
  }

  const TCopyParams getCopyParams() const {
    TCopyParams copy_params;
    copy_params.source_type = TSourceType::GEO_FILE;
    return copy_params;
  }

  TColumnType getScalarColumnType(const std::string& name,
                                  const TDatumType::type type) const {
    TColumnType ct;
    ct.col_name = name;
    ct.col_type.type = type;
    ct.col_type.encoding = TEncodingType::type::NONE;
    ct.col_type.nullable = true;
    ct.col_type.is_array = false;
    ct.col_type.precision = 0;
    ct.col_type.scale = 0;
    ct.col_type.comp_param = 0;
    ct.col_type.size = 0;
    ct.is_reserved_keyword = false;
    ct.src_name = name;
    ct.is_system = false;
    ct.is_physical = false;
    ct.col_id = 0;
    return ct;
  };

  TColumnType getPolyColumnType(const std::string& name) const {
    TColumnType ct;
    ct.col_name = name;
    ct.col_type.type = TDatumType::type::POLYGON;
    ct.col_type.encoding = TEncodingType::type::GEOINT;
    ct.col_type.nullable = true;
    ct.col_type.is_array = false;
    ct.col_type.precision = 23;  // aka subtype = SQLTypes::kGEOMETRY (not the same as
                                 // TDatumType::type::GEOMETRY)
    ct.col_type.scale = 4326;    // aka output_srid = WGS84
    ct.col_type.comp_param = 32;
    ct.col_type.size = 0;
    ct.is_reserved_keyword = false;
    ct.src_name = name;
    ct.is_system = false;
    ct.is_physical = false;
    ct.col_id = 0;
    return ct;
  };

  void TearDown() override {
    sql("DROP TABLE IF EXISTS import_geo_table_test");
    SqlTestFixture::TearDown();
  }
};

TEST_F(ImportGeoTableTest, ImportGeoTableAuto) {
  // geo import with empty row descriptor
  // will automatically create table
  // equivalent to COPY FROM WITH (source_type='geo_file')
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRowDescriptor row_descriptor;
  EXPECT_NO_THROW(handler->import_geo_table(session,
                                            "import_geo_table_test",
                                            getGeoFileName(),
                                            getCopyParams(),
                                            row_descriptor));
  sqlAndCompareResult("SELECT count(*) FROM import_geo_table_test", {{i(10)}});
  sqlAndCompareResult("SELECT trip FROM import_geo_table_test WHERE rowid=0", {{0.0f}});
}

TEST_F(ImportGeoTableTest, ImportGeoTableExplicit) {
  // geo import with explicit row descriptor (e.g. Immerse import)
  // must create table first
  // correct types
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRowDescriptor row_descriptor{getScalarColumnType("trip", TDatumType::type::FLOAT),
                                getPolyColumnType(Geospatial::kGeoColumnName)};
  EXPECT_NO_THROW(
      handler->create_table(session, "import_geo_table_test", row_descriptor));
  EXPECT_NO_THROW(handler->import_geo_table(session,
                                            "import_geo_table_test",
                                            getGeoFileName(),
                                            getCopyParams(),
                                            row_descriptor));
  sqlAndCompareResult("SELECT count(*) FROM import_geo_table_test", {{i(10)}});
  sqlAndCompareResult("SELECT trip FROM import_geo_table_test WHERE rowid=0", {{0.0f}});
}

TEST_F(ImportGeoTableTest, ImportGeoTableOverride) {
  // geo import with explicit row descriptor (e.g. Immerse import)
  // must create table first
  // type of column 'trip' overridden from FLOAT to INT (valid)
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRowDescriptor row_descriptor{getScalarColumnType("trip", TDatumType::type::INT),
                                getPolyColumnType(Geospatial::kGeoColumnName)};
  EXPECT_NO_THROW(
      handler->create_table(session, "import_geo_table_test", row_descriptor));
  EXPECT_NO_THROW(handler->import_geo_table(session,
                                            "import_geo_table_test",
                                            getGeoFileName(),
                                            getCopyParams(),
                                            row_descriptor));
  sqlAndCompareResult("SELECT count(*) FROM import_geo_table_test", {{i(10)}});
  sqlAndCompareResult("SELECT trip FROM import_geo_table_test WHERE rowid=0", {{i(0)}});
}

TEST_F(ImportGeoTableTest, ImportGeoTableTypeMismatch1) {
  // geo import with explicit row descriptor (e.g. Immerse import)
  // must create table first
  // types of columns swapped (possible in Immerse for now)
  // import will not fail, but should reject all rows
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRowDescriptor row_descriptor{
      getPolyColumnType("trip"),
      getScalarColumnType(Geospatial::kGeoColumnName, TDatumType::type::FLOAT)};
  EXPECT_NO_THROW(
      handler->create_table(session, "import_geo_table_test", row_descriptor));
  EXPECT_THROW(handler->import_geo_table(session,
                                         "import_geo_table_test",
                                         getGeoFileName(),
                                         getCopyParams(),
                                         row_descriptor),
               TDBException);
  sqlAndCompareResult("SELECT count(*) FROM import_geo_table_test", {{i(0)}});
}

TEST_F(ImportGeoTableTest, ImportGeoTableFailTypeMismatch2) {
  // geo import with explicit row descriptor (e.g. Immerse import)
  // must create table first
  // column types valid but columns swapped (possible in Immerse for now)
  // import will not fail, but should reject all rows
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRowDescriptor row_descriptor{
      getScalarColumnType(Geospatial::kGeoColumnName, TDatumType::type::FLOAT),
      getPolyColumnType("trip")};
  EXPECT_NO_THROW(
      handler->create_table(session, "import_geo_table_test", row_descriptor));
  EXPECT_THROW(handler->import_geo_table(session,
                                         "import_geo_table_test",
                                         getGeoFileName(),
                                         getCopyParams(),
                                         row_descriptor),
               TDBException);
  sqlAndCompareResult("SELECT count(*) FROM import_geo_table_test", {{i(0)}});
}

TEST_F(ImportGeoTableTest, ImportGeoTableFailNoGeoColumns) {
  // geo import with explicit row descriptor (e.g. Immerse import)
  // must create table first
  // no geo columns in row descriptor
  // import should fail
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRowDescriptor row_descriptor{getScalarColumnType("trip", TDatumType::type::FLOAT)};
  EXPECT_NO_THROW(
      handler->create_table(session, "import_geo_table_test", row_descriptor));
  EXPECT_THROW(handler->import_geo_table(session,
                                         "import_geo_table_test",
                                         getGeoFileName(),
                                         getCopyParams(),
                                         row_descriptor),
               TDBException);
  sqlAndCompareResult("SELECT count(*) FROM import_geo_table_test", {{i(0)}});
}

TEST_F(ImportGeoTableTest, ImportGeoTableFailTooManyGeoColumns) {
  // geo import with explicit row descriptor (e.g. Immerse import)
  // must create table first
  // more than one geo column in row descriptor
  // import should fail
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TRowDescriptor row_descriptor{getScalarColumnType("trip", TDatumType::type::FLOAT),
                                getPolyColumnType("geo1"),
                                getPolyColumnType("geo2")};
  EXPECT_NO_THROW(
      handler->create_table(session, "import_geo_table_test", row_descriptor));
  EXPECT_THROW(handler->import_geo_table(session,
                                         "import_geo_table_test",
                                         getGeoFileName(),
                                         getCopyParams(),
                                         row_descriptor),
               TDBException);
  sqlAndCompareResult("SELECT count(*) FROM import_geo_table_test", {{i(0)}});
}


class ThriftFileGlobbingTest : public SqlTestFixture,
                               public testing::WithParamInterface<std::string> {
 protected:
  void SetUp() override {
    SqlTestFixture::SetUp();
    sql("DROP TABLE IF EXISTS " + test_table_name_ + ";");
    createTable();
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS " + test_table_name_ + ";");
    if (boost::filesystem::exists(empty_dir_path_)) {
      boost::filesystem::remove_all(empty_dir_path_);
    }
    SqlTestFixture::TearDown();
  }

  void createTable() {
    if (isGeoImport()) {
      sql("CREATE TABLE " + test_table_name_ + " (trip DOUBLE, geom POINT);");
    } else if (isRasterImport()) {
      sql("CREATE TABLE " + test_table_name_ +
          " (raster_lon DOUBLE, raster_lat DOUBLE, band_1_1 INTEGER);");
    } else {
      sql("CREATE TABLE " + test_table_name_ + " (i INTEGER);");
    }
  }

  bool isGeoImport() { return GetParam() == "geo"; }

  bool isRasterImport() { return GetParam() == "raster"; }

  bool isParquetImport() { return GetParam() == "parquet"; }

  bool isDelimitedImport() { return GetParam() == "delimited"; }

  void assertDetectResult(const TDetectResult& result) {
    const auto& row_set = result.row_set;
    if (isGeoImport()) {
      ASSERT_EQ(row_set.row_desc.size(), static_cast<size_t>(2));
      ASSERT_EQ(row_set.row_desc[0].col_name, "trip");
      ASSERT_EQ(row_set.row_desc[0].col_type.type, TDatumType::DOUBLE);
      ASSERT_EQ(row_set.row_desc[1].col_name, "geom");
      ASSERT_EQ(row_set.row_desc[1].col_type.type, TDatumType::POINT);
      ASSERT_EQ(row_set.rows.size(), static_cast<size_t>(10));
      for (const auto& row : row_set.rows) {
        ASSERT_EQ(row.cols.size(), static_cast<size_t>(2));
      }
    } else if (isRasterImport()) {
      ASSERT_EQ(row_set.row_desc.size(), static_cast<size_t>(3));
      ASSERT_EQ(row_set.row_desc[0].col_name, "raster_lon");
      ASSERT_EQ(row_set.row_desc[0].col_type.type, TDatumType::DOUBLE);
      ASSERT_EQ(row_set.row_desc[1].col_name, "raster_lat");
      ASSERT_EQ(row_set.row_desc[1].col_type.type, TDatumType::DOUBLE);
      ASSERT_EQ(row_set.row_desc[2].col_name, "band_1_1");
      ASSERT_EQ(row_set.row_desc[2].col_type.type, TDatumType::INT);
      ASSERT_TRUE(row_set.rows.empty());
    } else {
      CHECK(isParquetImport() || isDelimitedImport());
      ASSERT_EQ(row_set.row_desc.size(), static_cast<size_t>(1));
      ASSERT_EQ(row_set.row_desc[0].col_name, "i");
      if (isParquetImport()) {
        ASSERT_EQ(row_set.row_desc[0].col_type.type, TDatumType::BIGINT);
      } else {
        ASSERT_EQ(row_set.row_desc[0].col_type.type, TDatumType::SMALLINT);
      }
      ASSERT_EQ(row_set.rows.size(), static_cast<size_t>(2));
      ASSERT_EQ(row_set.rows[0].cols.size(), static_cast<size_t>(1));
      ASSERT_EQ(row_set.rows[0].cols[0].val.str_val, "2");
      ASSERT_EQ(row_set.rows[1].cols.size(), static_cast<size_t>(1));
      ASSERT_EQ(row_set.rows[1].cols[0].val.str_val, "1");
    }
  }

  TCopyParams getCopyParams() {
    TCopyParams copy_params;
    if (isGeoImport()) {
      copy_params.source_type = TSourceType::GEO_FILE;
    } else if (isRasterImport()) {
      copy_params.source_type = TSourceType::RASTER_FILE;
    } else if (isParquetImport()) {
      copy_params.source_type = TSourceType::PARQUET_FILE;
    } else if (isDelimitedImport()) {
      copy_params.source_type = TSourceType::DELIMITED_FILE;
    } else {
      UNREACHABLE() << "Unexpected import type: " << GetParam();
    }
    return copy_params;
  }

  std::string getTestGlobPath() {
    auto path = boost::filesystem::canonical("../../Tests").string();
    if (isGeoImport()) {
      path += "/Import/datafiles/geospatial_point/geospatial_point_*.geojson";
    } else if (isRasterImport()) {
      path += "/Import/datafiles/raster/s1b_*.tiff";
    } else if (isParquetImport()) {
      path += "/FsiDataFiles/sorted_dir/parquet/*1.parquet";
    } else if (isDelimitedImport()) {
      path += "/FsiDataFiles/sorted_dir/csv/*1.csv";
    } else {
      UNREACHABLE() << "Unexpected import type: " << GetParam();
    }
    return path;
  }

  static std::string getNonExistentGlobPath() {
    return boost::filesystem::canonical("../../Tests/FsiDataFiles/sorted_dir").string() +
           "/nonexistent*";
  }

  static std::string getEmptyDirectoryPath() {
    return boost::filesystem::canonical(empty_dir_path_).string();
  }

  void importTable(const std::string& file_path) {
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    if (isGeoImport() || isRasterImport()) {
      db_handler->import_geo_table(
          session_id, test_table_name_, file_path, getCopyParams(), {});
    } else {
      CHECK(isParquetImport() || isDelimitedImport());
      db_handler->import_table(session_id, test_table_name_, file_path, getCopyParams());
    }
  }

  int64_t getTestImportRowCount() {
    if (isGeoImport()) {
      return 30;
    } else if (isRasterImport()) {
      return 40000;
    } else {
      CHECK(isParquetImport() || isDelimitedImport());
      return 2;
    }
  }

  inline static const std::string test_table_name_{"test_table"};
  inline static const std::string empty_dir_path_{"./test_import_empty_dir"};
};

TEST_P(ThriftFileGlobbingTest, Detect) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TDetectResult result;
  db_handler->detect_column_types(result, session_id, getTestGlobPath(), getCopyParams());
  assertDetectResult(result);
}

TEST_P(ThriftFileGlobbingTest, DetectWithNonExistentGlobPath) {
  executeLambdaAndAssertException(
      [this] {
        const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
        TDetectResult result;
        db_handler->detect_column_types(
            result, session_id, getNonExistentGlobPath(), getCopyParams());
      },
      "File or directory \"" + getNonExistentGlobPath() + "\" does not exist.");
}

TEST_P(ThriftFileGlobbingTest, Import) {
  sqlAndCompareResult("SELECT * FROM " + test_table_name_ + ";", {});
  importTable(getTestGlobPath());
  sqlAndCompareResult("SELECT COUNT(*) FROM " + test_table_name_ + ";",
                      {{getTestImportRowCount()}});
  if (isParquetImport() || isDelimitedImport()) {
    sqlAndCompareResult("SELECT * FROM " + test_table_name_ + " ORDER BY i;",
                        {{i(1)}, {i(2)}});
  }
}

TEST_P(ThriftFileGlobbingTest, ImportWithNonExistentGlobPath) {
  executeLambdaAndAssertException(
      [this] { importTable(getNonExistentGlobPath()); },
      "File or directory \"" + getNonExistentGlobPath() + "\" does not exist.");
}

TEST_P(ThriftFileGlobbingTest, DetectEmptyDirectory) {
  boost::filesystem::create_directory(empty_dir_path_);
  std::string error_message{"detect_column_types error: "};
  if (isGeoImport()) {
    error_message +=
        "openGDALDataSource Error: Unable to open geo file " + getEmptyDirectoryPath();
  } else if (isRasterImport()) {
    error_message +=
        "Raster Importer: Unable to open raster file " + getEmptyDirectoryPath();
  } else if (isParquetImport()) {
    error_message += "No file found at \"" + getEmptyDirectoryPath() + "\"";
  } else if (isDelimitedImport()) {
    error_message += "No rows found in: test_import_empty_dir";
  } else {
    UNREACHABLE() << "Unexpected import type: " << GetParam();
  }
  executeLambdaAndAssertException(
      [this] {
        const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
        TDetectResult result;
        db_handler->detect_column_types(
            result, session_id, getEmptyDirectoryPath(), getCopyParams());
      },
      error_message);
}

INSTANTIATE_TEST_SUITE_P(DifferentImportTypes,
                         ThriftFileGlobbingTest,
                         testing::Values("delimited", "parquet", "geo", "raster"),
                         [](const auto& param_info) { return param_info.param; });

class ThriftDetectGeoFileTest : public SqlTestFixture {
 protected:
  const TCopyParams getCopyParams() const {
    TCopyParams copy_params;
    copy_params.source_type = TSourceType::GEO_FILE;
    return copy_params;
  }

  const std::string getTestFile() const {
    return boost::filesystem::canonical(
               "../../Tests/ImportGeoTableTest/datafiles/text_column.geojson")
        .string();
  }

  void assertDetectResult(const TDetectResult& result) {
    const auto& row_set = result.row_set;
    ASSERT_EQ(row_set.row_desc.size(), static_cast<size_t>(3));
    for (const auto& row : row_set.rows) {
      ASSERT_EQ(row.cols.size(), static_cast<size_t>(3));
    }

    ASSERT_EQ(row_set.row_desc[0].col_name, "test");
    ASSERT_EQ(row_set.row_desc[0].col_type.type, TDatumType::STR);
    ASSERT_EQ(row_set.row_desc[0].col_type.encoding, TEncodingType::type::DICT);
    ASSERT_EQ(row_set.row_desc[0].col_type.comp_param, 0);

    ASSERT_EQ(row_set.row_desc[1].col_name, "test_mls");
    ASSERT_EQ(row_set.row_desc[1].col_type.type, TDatumType::STR);
    ASSERT_EQ(row_set.row_desc[1].col_type.encoding, TEncodingType::type::DICT);
    ASSERT_EQ(row_set.row_desc[1].col_type.comp_param, 0);

    ASSERT_EQ(row_set.row_desc[2].col_name, "geom");
    ASSERT_EQ(row_set.row_desc[2].col_type.type, TDatumType::MULTILINESTRING);
    ASSERT_EQ(row_set.row_desc[2].col_type.encoding, TEncodingType::type::GEOINT);
    ASSERT_EQ(row_set.row_desc[2].col_type.comp_param, 32);
  }
};

TEST_F(ThriftDetectGeoFileTest, EncodedTextColumnUsesTransientDictionary) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TDetectResult result;
  db_handler->detect_column_types(result, session_id, getTestFile(), getCopyParams());
  assertDetectResult(result);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    SqlTestFixture::initConnection(argc, argv);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = 1;
  }
  return err;
}
