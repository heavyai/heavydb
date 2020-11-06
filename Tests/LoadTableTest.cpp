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

#include <gtest/gtest.h>

#include "Tests/DBHandlerTestHelpers.h"
#include "Tests/TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class LoadTableTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("DROP TABLE IF EXISTS load_test");
    sql("DROP TABLE IF EXISTS geo_load_test");
    sql("CREATE TABLE geo_load_test(i1 INTEGER, ls LINESTRING, s TEXT, mp MULTIPOLYGON, "
        "nns TEXT not null)");
    sql("CREATE TABLE load_test(i1 INTEGER, s TEXT, nns TEXT not null)");
    initData();
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS load_test");
    sql("DROP TABLE IF EXISTS geo_load_test");
    DBHandlerTestFixture::TearDown();
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
  const std::string MULTIPOLYGON =
      "MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0),"
      "(1 1,1 2,2 2,2 1,1 1)),((-1 -1,-2 -1,-2 -2,-1 -2,-1 -1)))";
  TColumn i1_column, s_column, nns_column, ls_column, mp_column;

 private:
  void initData() {
    i1_column.nulls = s_column.nulls = nns_column.nulls = ls_column.nulls =
        mp_column.nulls = {false};
    i1_column.data.int_col = {1};
    s_column.data.str_col = {"s"};
    nns_column.data.str_col = {"nns"};
    ls_column.data.str_col = {LINESTRING};
    mp_column.data.str_col = {MULTIPOLYGON};
  }
};

TEST_F(LoadTableTest, AllColumnsNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TStringRow row;
  row.cols = {getSV("1"), getSV("s"), getSV("nns")};
  handler->load_table(session, "load_test", {row});
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "s", "nns"}});
}

TEST_F(LoadTableTest, AllColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TStringRow row;
  row.cols = {
      getSV("1"), getSV(LINESTRING), getSV("s"), getSV(MULTIPOLYGON), getSV("nns")};
  handler->load_table(session, "geo_load_test", {row});
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", MULTIPOLYGON, "nns"}});
}

TEST_F(LoadTableTest, NullGeoColumn) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TStringRow row;
  row.cols = {getSV("1"), getSV(LINESTRING), getSV("s"), getNullSV(), getSV("nns")};
  handler->load_table(session, "geo_load_test", {row});
  sqlAndCompareResult("SELECT i1, s, nns, mp, ls FROM geo_load_test",
                      {{i(1), "s", "nns", "NULL", LINESTRING}});
}

TEST_F(LoadTableTest, ColumnarAllColumnsNoGeo) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  handler->load_table_binary_columnar(
      session, "load_test", {i1_column, s_column, nns_column});
  sqlAndCompareResult("SELECT * FROM load_test", {{i(1), "s", "nns"}});
}

TEST_F(LoadTableTest, ColumnarAllColumns) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  handler->load_table_binary_columnar(
      session, "geo_load_test", {i1_column, ls_column, s_column, mp_column, nns_column});
  sqlAndCompareResult("SELECT * FROM geo_load_test",
                      {{i(1), LINESTRING, "s", MULTIPOLYGON, "nns"}});
}

TEST_F(LoadTableTest, ColumnarNullGeoColumn) {
  auto* handler = getDbHandlerAndSessionId().first;
  auto& session = getDbHandlerAndSessionId().second;
  TColumn null_column;
  null_column.nulls = {false};
  null_column.data.str_col = {""};
  handler->load_table_binary_columnar(
      session,
      "geo_load_test",
      {i1_column, ls_column, s_column, null_column, nns_column});
  sqlAndCompareResult("SELECT i1, s, nns, mp, ls FROM geo_load_test",
                      {{i(1), "s", "nns", "NULL", LINESTRING}});
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
