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

#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include "boost/filesystem.hpp"

#include "Catalog/Catalog.h"
#include "DBHandlerTestHelpers.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Geospatial/Types.h"
#include "ImportExport/Importer.h"
#include "Parser/parser.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/UpdelRoll.h"
#include "Shared/scope.h"
#include "Tests/TestHelpers.h"

#include <tuple>
#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace Catalog_Namespace;
using namespace TestHelpers;

using QR = QueryRunner::QueryRunner;

extern bool g_test_drop_column_rollback;

namespace {

bool g_hoist_literals{true};

inline void run_ddl_statement(const std::string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

std::shared_ptr<ResultSet> run_query(const std::string& query_str) {
  return QR::get()->runSQL(query_str, ExecutorDeviceType::CPU, g_hoist_literals, true);
}

std::unique_ptr<QR> get_qr_for_user(
    const std::string& user_name,
    const Catalog_Namespace::UserMetadata& user_metadata) {
  auto session = std::make_unique<Catalog_Namespace::SessionInfo>(
      Catalog_Namespace::SysCatalog::instance().getCatalog(user_name),
      user_metadata,
      ExecutorDeviceType::CPU,
      "");
  return std::make_unique<QR>(std::move(session));
}

template <typename E = std::runtime_error>
bool alter_common(const std::string& table,
                  const std::string& column,
                  const std::string& type,
                  const std::string& comp,
                  const std::string& val,
                  const std::string& val2,
                  const bool expect_throw = false) {
  std::string alter_query = "alter table " + table + " add column " + column + " " + type;
  if (val != "") {
    alter_query += " default " + val;
  }
  if (comp != "") {
    alter_query += " encoding " + comp;
  }

  if (expect_throw) {
    EXPECT_THROW(run_ddl_statement(alter_query + ";"), E);
    return true;
  } else {
    EXPECT_NO_THROW(run_ddl_statement(alter_query + ";"););
  }

  if (val2 != "") {
    std::string query_str = "SELECT " + column + " FROM " + table;
    auto rows = run_query(query_str + ";");
    int r_cnt = 0;
    while (true) {
      auto crt_row = rows->getNextRow(true, true);
      if (0 == crt_row.size()) {
        break;
      }
      auto geo = v<NullableString>(crt_row[0]);
      auto geo_s = boost::get<std::string>(&geo);
      auto geo_v = boost::get<void*>(&geo);

#if 1
      if (!geo_s && geo_v && *geo_v == nullptr && val2 == "NULL") {
        ++r_cnt;
      }
      if (!geo_v && geo_s && *geo_s == val2) {
        ++r_cnt;
      }
#else
      // somehow these do not work as advertised ...
      using namespace Geospatial;
      if (boost::iequals(type, "POINT") && GeoPoint(geo) == GeoPoint(val2))
        ++r_cnt;
      else if (boost::iequals(type, "LINESTRING") &&
               GeoLineString(geo) == GeoLineString(val2))
        ++r_cnt;
      else if (boost::iequals(type, "POLYGON") && GeoPolygon(geo) == GeoPolygon(val2))
        ++r_cnt;
      else if (boost::iequals(type, "MULTIPOLYGON") &&
               GeoMultiPolygon(geo) == GeoMultiPolygon(val2))
        ++r_cnt;
#endif
    }
    return r_cnt == 100;
  } else {
    std::string query_str =
        "SELECT count(*) FROM " + table + " WHERE " + column +
        ("" == val || boost::iequals("NULL", val) ? " IS NULL" : (" = " + val));
    auto rows = run_query(query_str + ";");
    auto crt_row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(1), crt_row.size());
    auto r_cnt = v<int64_t>(crt_row[0]);
    return r_cnt == 100;
  }
}

void import_table_file(const std::string& table, const std::string& file) {
  const auto query_str = std::string("COPY trips FROM '") +
                         "../../Tests/Import/datafiles/" + file +
                         "' WITH (header='true');";

  auto stmt = QR::get()->createDDLStatement(query_str);

  auto copy_stmt = dynamic_cast<Parser::CopyTableStmt*>(stmt.get());
  if (!copy_stmt) {
    throw std::runtime_error("Expected a CopyTableStatment: " + query_str);
  }
  QR::get()->runImport(copy_stmt);
}

// don't use R"()" format; somehow it causes many blank lines
// to be output on console. how come?
const char* create_table_trips =
    "    CREATE TABLE trips ("
    "            medallion               TEXT ENCODING DICT,"
    "            hack_license            TEXT ENCODING DICT,"
    "            vendor_id               TEXT ENCODING DICT,"
    "            rate_code_id            SMALLINT,"
    "            store_and_fwd_flag      TEXT ENCODING DICT,"
    "            pickup_datetime         TIMESTAMP,"
    "            dropoff_datetime        TIMESTAMP,"
    "            passenger_count         SMALLINT,"
    "            trip_time_in_secs       INTEGER,"
    "            trip_distance           FLOAT,"
    "            pickup_longitude        DECIMAL(14,7),"
    "            pickup_latitude         DECIMAL(14,7),"
    "            dropoff_longitude       DOUBLE,"
    "            dropoff_latitude        DECIMAL(18,5),"
    "            deleted                 BOOLEAN"
    "            ) WITH (FRAGMENT_SIZE=50);";  // so 2 fragments here

void init_table_data(const std::string& table = "trips",
                     const std::string& create_table_cmd = create_table_trips,
                     const std::string& file = "trip_data_dir/trip_data_b.txt") {
  run_ddl_statement("drop table if exists " + table + ";");
  run_ddl_statement(create_table_cmd);
  if (file.size()) {
    import_table_file(table, file);
  }
}

class AlterColumnTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_NO_THROW(init_table_data();); }
  void TearDown() override { ASSERT_NO_THROW(run_ddl_statement("drop table trips;");); }
};

#define MT std::make_tuple
std::vector<std::tuple<std::string, std::string, std::string, std::string>> type_vals = {
    MT("text", "none", "'abc'", ""),
    MT("text", "dict(8)", "'ijk'", ""),
    MT("text", "dict(32)", "'xyz'", ""),
    MT("float", "", "1.25", ""),
    MT("double", "", "1.25", ""),
    MT("smallint", "", "123", ""),
    MT("integer", "", "123", ""),
    MT("bigint", "", "123", ""),
    MT("bigint encoding fixed(8)", "", "", ""),
    MT("bigint encoding fixed(16)", "", "", ""),
    MT("bigint encoding fixed(32)", "", "", ""),
    MT("decimal(8)", "", "123", ""),
    MT("decimal(8,2)", "", "1.23", ""),
    MT("date", "", "'2011-10-23'", ""),
    MT("time", "", "'10:23:45'", ""),
    MT("timestamp", "", "'2011-10-23 10:23:45'", ""),
    MT("POINT", "", "'POINT (1 2)'", "POINT (1 2)"),
    MT("LINESTRING", "", "'LINESTRING (1 1,2 2,3 3)'", "LINESTRING (1 1,2 2,3 3)"),
    MT("POLYGON",
       "",
       "'POLYGON((0 0,0 9,9 9,9 0),(1 1,2 2,3 3))'",
       "POLYGON ((9 0,9 9,0 9,0 0,9 0),(3 3,2 2,1 1,3 3))"),
    MT("MULTIPOLYGON",
       "",
       "'MULTIPOLYGON(((0 0,0 9,9 9,9 0),(1 1,2 2,3 3)))'",
       "MULTIPOLYGON (((9 0,9 9,0 9,0 0,9 0),(3 3,2 2,1 1,3 3)))"),
};
#undef MT

TEST_F(AlterColumnTest, Add_column_with_default) {
  int cid = 0;
  for (const auto& tv : type_vals) {
    EXPECT_TRUE(alter_common("trips",
                             "x" + std::to_string(++cid),
                             std::get<0>(tv),
                             std::get<1>(tv),
                             std::get<2>(tv),
                             std::get<3>(tv),
                             false));
  }
}

TEST_F(AlterColumnTest, Add_column_with_null) {
  int cid = 0;
  for (const auto& tv : type_vals) {
    if (std::get<3>(tv) == "") {
      EXPECT_TRUE(alter_common("trips",
                               "x" + std::to_string(++cid),
                               std::get<0>(tv),
                               std::get<1>(tv),
                               "",
                               "",
                               false));
    } else {
      // Geometry column
      // Doesn't throw, no explicit default (default is NULL geo),
      EXPECT_TRUE(alter_common("trips",
                               "x" + std::to_string(++cid),
                               std::get<0>(tv),
                               std::get<1>(tv),
                               "",  // no explicit default (NULL geo)
                               "NULL",
                               false));
    }
  }
}

TEST(AlterColumnTest2, Drop_after_fail_to_add) {
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists t;"););
  EXPECT_NO_THROW(run_ddl_statement("create table t(c1 int);"););
  EXPECT_NO_THROW(run_query("insert into t values (10);"););
  EXPECT_THROW(
      run_ddl_statement("alter table t add column c2 TEXT NOT NULL ENCODING DICT;"),
      std::runtime_error);
  EXPECT_NO_THROW(run_ddl_statement("drop table t;"););
}

TEST(AlterColumnTest3, Add_col_to_sharded_table) {
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists x;"););
  EXPECT_NO_THROW(run_ddl_statement(
                      "create table x (i text,SHARD KEY (i)) WITH (SHARD_COUNT = 2);"););
  EXPECT_NO_THROW(run_ddl_statement("alter table x add column j int;"););
  EXPECT_NO_THROW(run_query("insert into x values('0',0);"););
}

void drop_columns(const bool rollback, const std::vector<std::string>&& dropped_columns) {
  g_test_drop_column_rollback = rollback;
  std::vector<std::string> drop_column_phrases;
  std::transform(
      dropped_columns.begin(),
      dropped_columns.end(),
      std::back_inserter(drop_column_phrases),
      [](const auto& dropped_column) -> auto {
        using namespace std::string_literals;
        return dropped_column;
      });
  std::string drop_column_statement = "alter table t drop column " +
                                      boost::algorithm::join(drop_column_phrases, ",") +
                                      ";";
  if (g_test_drop_column_rollback) {
    EXPECT_THROW(run_ddl_statement(drop_column_statement), std::runtime_error);
    for (const auto& dropped_column : dropped_columns) {
      EXPECT_NO_THROW(run_query("select count(" + dropped_column + ") from t;"));
    }
  } else {
    EXPECT_NO_THROW(run_ddl_statement(drop_column_statement););
    for (const auto& dropped_column : dropped_columns) {
      EXPECT_THROW(run_query("select count(" + dropped_column + ") from t;"),
                   std::exception);
    }
  }
  const auto rows = run_query("select count(a), count(f) from t;");
  const auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  CHECK_EQ(v<int64_t>(crt_row[0]), 2);
  CHECK_EQ(v<int64_t>(crt_row[1]), 2);
  EXPECT_NO_THROW(run_query("select a from t;"));
}

class AlterColumnTest4 : public ::testing::Test {
 protected:
  void SetUp() override {
    EXPECT_NO_THROW(run_ddl_statement("drop view if exists v;"););
    EXPECT_NO_THROW(run_ddl_statement("drop table if exists t;"););
    EXPECT_NO_THROW(
        run_ddl_statement("create table t(a text, t text, b int, c point, shared "
                          "dictionary(t) references t(a)) with (fragment_size=1);"););
    EXPECT_NO_THROW(run_ddl_statement("alter table t add d point;"););
    EXPECT_NO_THROW(run_ddl_statement("alter table t add e int;"););
    EXPECT_NO_THROW(run_ddl_statement("alter table t add f float;"););
    EXPECT_NO_THROW(
        run_query(
            "insert into t values ('0', '0', 0, 'point(0 0)', 'point(0 0)', 0, 0);"););
    EXPECT_NO_THROW(
        run_query(
            "insert into t values ('1', '1', 1, 'point(1 1)', 'point(1 1)', 1, 1);"););
    g_test_drop_column_rollback = false;
  }
  void TearDown() override { EXPECT_NO_THROW(run_ddl_statement("drop table t;");); }
};

TEST_F(AlterColumnTest4, Consecutive_drop_columns_different_data_types) {
  drop_columns(false, {"b"});
  drop_columns(false, {"c", "d"});
  drop_columns(false, {"e", "t"});
}

TEST_F(AlterColumnTest4, Drop_columns_rollback) {
  drop_columns(true, {"a", "c", "d", "f"});
}

TEST_F(AlterColumnTest4, Drop_column_referenced_by_view) {
  EXPECT_NO_THROW(run_ddl_statement("create view v as select b from t;"););
  drop_columns(false, {"b"});
  EXPECT_THROW(run_query("select count(b) from v;"), std::exception);
}

TEST_F(AlterColumnTest4, Alter_column_of_view) {
  EXPECT_NO_THROW(run_ddl_statement("create view v as select b from t;"););
  EXPECT_THROW(run_ddl_statement("alter table v add column i int;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("alter table v drop column b;"), std::runtime_error);
}

TEST_F(AlterColumnTest4, Alter_inexistent_table) {
  EXPECT_THROW(run_ddl_statement("alter table xx drop column xxx;"), std::runtime_error);
}

TEST_F(AlterColumnTest4, Alter_inexistent_table_column) {
  EXPECT_THROW(run_ddl_statement("alter table t drop column xxx;"), std::runtime_error);
}

TEST(AlterColumnTest5, Drop_the_only_column) {
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists x;"););
  EXPECT_NO_THROW(run_ddl_statement("create table x (i int);"););
  EXPECT_THROW(run_ddl_statement("alter table x drop column i;"), std::runtime_error);
}

TEST(AlterColumnTest5, Drop_sharding_column) {
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists x;"););
  EXPECT_NO_THROW(
      run_ddl_statement(
          "create table x (i int, j int, SHARD KEY (i)) WITH (SHARD_COUNT = 2);"););
  EXPECT_THROW(run_ddl_statement("alter table x drop column i;"), std::runtime_error);
}

TEST(AlterColumnTest5, DISABLED_Drop_temp_table_column) {
  // TODO(adb): The Catalog still runs SQLite queries with drop column. While they are
  // essentially no-op queries, we should disable running the queries for both alter and
  // drop in a consistent way. Currently Alter/drop are disabled on temp tables.
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists x;"););
  EXPECT_NO_THROW(run_ddl_statement("create TEMPORARY table x (i int, j int);"););
  EXPECT_NO_THROW(run_query("insert into x values (0,0);"););
  EXPECT_NO_THROW(run_query("insert into x values (1,1);"););
  EXPECT_NO_THROW(run_ddl_statement("alter table x drop column j;"););
  EXPECT_NO_THROW(run_query("select i from x;"));
}

TEST(AlterColumnTest5, Drop_table_by_unauthorized_user) {
  using namespace std::string_literals;
  auto admin = "admin"s;
  auto admin_password = "HyperInteractive"s;
  auto thief = "thief"s;
  auto thief_password = "thief"s;
  auto lucky = "lucky"s;
  auto lucky_password = "lucky"s;
  auto mydb = "mydb"s;
  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
  EXPECT_NO_THROW(run_ddl_statement("CREATE DATABASE mydb (owner='admin');"););
  EXPECT_NO_THROW(run_ddl_statement("CREATE USER thief (password='thief');"););
  EXPECT_NO_THROW(run_ddl_statement("CREATE USER lucky (password='lucky');"););
  ScopeGuard scope_guard = [] {
    run_ddl_statement("DROP USER lucky;");
    run_ddl_statement("DROP USER thief;");
    run_ddl_statement("DROP DATABASE mydb;");
  };
  // login to mydb as admin
  Catalog_Namespace::UserMetadata user_meta1;
  EXPECT_NO_THROW(sys_cat.login(mydb, admin, admin_password, user_meta1, false););
  auto qr1 = get_qr_for_user(mydb, user_meta1);
  auto dt = ExecutorDeviceType::CPU;
  EXPECT_NO_THROW(qr1->runDDLStatement("CREATE TABLE x (i int, j int, k int);"));
  EXPECT_NO_THROW(qr1->runSQL("insert into x values (0, 0, 0);", dt));
  EXPECT_NO_THROW(qr1->runDDLStatement("alter table x drop column k;"));
  EXPECT_NO_THROW(qr1->runSQL("select i from x;", dt));
  EXPECT_NO_THROW(qr1->runDDLStatement("grant alter on table x to lucky;"));
  // login to mydb as thief
  Catalog_Namespace::UserMetadata user_meta2;
  EXPECT_NO_THROW(sys_cat.login(mydb, thief, thief_password, user_meta2, false));
  auto qr2 = get_qr_for_user(mydb, user_meta2);
  EXPECT_THROW(qr2->runDDLStatement("alter table x drop column j;"), std::runtime_error);
  // login to mydb as lucky
  Catalog_Namespace::UserMetadata user_meta3;
  EXPECT_NO_THROW(sys_cat.login(mydb, lucky, lucky_password, user_meta3, false));
  auto qr3 = get_qr_for_user(mydb, user_meta3);
  EXPECT_NO_THROW(qr3->runDDLStatement("alter table x drop column j;"));
}

}  // namespace

class AlterTableSetMaxRowsTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
  }

  void TearDown() override {
    sql("drop table if exists test_table;");
    DBHandlerTestFixture::TearDown();
  }

  void insertRange(size_t start, size_t end) {
    for (size_t i = start; i <= end; i++) {
      sql("insert into test_table values (" + std::to_string(i) + ");");
    }
  }

  void assertMaxRows(int64_t max_rows) {
    auto td = getCatalog().getMetadataForTable("test_table", false);
    ASSERT_EQ(max_rows, td->maxRows);
  }
};

TEST_F(AlterTableSetMaxRowsTest, MaxRowsLessThanTableRows) {
  sql("create table test_table (i integer) with (fragment_size = 2);");
  insertRange(1, 5);
  sqlAndCompareResult("select * from test_table;",
                      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
  sql("alter table test_table set max_rows = 4;");
  assertMaxRows(4);

  // Oldest fragment is deleted, so last 3 rows should remain.
  sqlAndCompareResult("select * from test_table;", {{i(3)}, {i(4)}, {i(5)}});
}

TEST_F(AlterTableSetMaxRowsTest, MaxRowsLessThanTableRowsAndSingleFragment) {
  sql("create table test_table (i integer) with (fragment_size = 10);");
  insertRange(1, 5);
  sqlAndCompareResult("select * from test_table;",
                      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
  sql("alter table test_table set max_rows = 4;");
  assertMaxRows(4);

  // max_rows should not delete the only fragment in a table
  sqlAndCompareResult("select * from test_table;",
                      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
}

TEST_F(AlterTableSetMaxRowsTest, MaxRowsGreaterThanTableRows) {
  sql("create table test_table (i integer) with (fragment_size = 2);");
  insertRange(1, 5);
  sqlAndCompareResult("select * from test_table;",
                      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
  sql("alter table test_table set max_rows = 10;");
  assertMaxRows(10);
  sqlAndCompareResult("select * from test_table;",
                      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
}

TEST_F(AlterTableSetMaxRowsTest, NegativeMaxRows) {
  sql("create table test_table (i integer) with (fragment_size = 2);");
  insertRange(1, 5);
  sqlAndCompareResult("select * from test_table;",
                      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
  queryAndAssertException("alter table test_table set max_rows = -1;",
                          "Max rows cannot be a negative number.");
  assertMaxRows(DEFAULT_MAX_ROWS);
  sqlAndCompareResult("select * from test_table;",
                      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}});
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

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
