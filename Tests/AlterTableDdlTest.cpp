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

#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include "boost/filesystem.hpp"

#include "Catalog/Catalog.h"
#include "DBHandlerTestHelpers.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Geospatial/Types.h"
#include "ImportExport/Importer.h"
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

  auto stmt = QR::get()->createStatement(query_str);

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
      [](const auto& dropped_column) -> auto{
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

TEST_F(AlterTableSetMaxRowsTest, EmptyTable) {
  sql("create table test_table (i integer);");
  sql("alter table test_table set max_rows = 10;");
  assertMaxRows(10);
  sqlAndCompareResult("select * from test_table;", {});
}

class AlterTableAlterColumnTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP TABLE IF EXISTS test_temp_table;");
    sql("DROP VIEW IF EXISTS test_view;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP TABLE IF EXISTS test_temp_table;");
    sql("DROP VIEW IF EXISTS test_view;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    DBHandlerTestFixture::TearDown();
  }

  static void createTestUser() {
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  }

  static void createTextTable(const std::vector<std::string>& column_names,
                              const std::vector<std::vector<std::string>>& values,
                              const std::string text_encoding = "ENCODING NONE",
                              const std::string table_name = "test_table",
                              const bool treat_null_value_as_string = false,
                              const std::string options_str = {}) {
    std::string table_schema = " index INT";
    for (const auto& column_name : column_names) {
      if (!table_schema.empty()) {
        table_schema += ", ";
      }
      table_schema += column_name + " TEXT " + text_encoding;
    }
    if (options_str.empty()) {
      sql("CREATE TABLE " + table_name + " (" + table_schema + ");");
    } else {
      sql("CREATE TABLE " + table_name + " (" + table_schema + ") WITH (" + options_str +
          ");");
    }
    std::string data_str{};
    int index = 1;
    for (const auto& row : values) {
      std::string data_row = std::to_string(index++);
      for (const auto& value : row) {
        if (!data_row.empty()) {
          data_row += ", ";
        }
        if (!treat_null_value_as_string && value == "NULL") {
          data_row += value;
        } else {
          data_row += "'" + value + "'";
        }
      }
      data_row = "(" + data_row + ")";
      if (!data_str.empty()) {
        data_str += ", ";
      }
      data_str += data_row;
    }
    sql("INSERT INTO " + table_name + " VALUES " + data_str + " ;");
  }

  std::list<const ColumnDescriptor*> getAllColumns(const std::string& table_name) {
    auto& catalog = getCatalog();
    auto tid = catalog.getTableId(table_name);
    CHECK(tid.has_value());
    return catalog.getAllColumnMetadataForTable(tid.value(), false, false, false);
  }

  void compareSchemaToReference(
      const std::string& table_name,
      const std::vector<std::pair<std::string, std::string>>& reference_schema) {
    std::string ref_table_schema;
    for (const auto& [column_name, column_type] : reference_schema) {
      if (!ref_table_schema.empty()) {
        ref_table_schema += ", ";
      }
      ref_table_schema += column_name + " " + column_type;
    }
    sql("DROP TABLE IF EXISTS reference_table;");
    sql("CREATE TABLE reference_table (" + ref_table_schema + ");");
    auto ref_cols = getAllColumns("reference_table");
    auto orig_cols = getAllColumns(table_name);
    ASSERT_EQ(ref_cols.size(), orig_cols.size());
    for (auto rcd : ref_cols) {
      auto cdit = std::find_if(
          orig_cols.begin(), orig_cols.end(), [&rcd](const ColumnDescriptor* cd) {
            return cd->columnName == rcd->columnName;
          });
      ASSERT_NE(cdit, orig_cols.end());
      auto comparison_result =
          ddl_utils::alter_column_utils::compare_column_descriptors(*cdit, rcd);
      ASSERT_TRUE(comparison_result.sql_types_match);
      ASSERT_TRUE(comparison_result.defaults_match);
    }
    sql("DROP TABLE reference_table;");
  }
};

TEST_F(AlterTableAlterColumnTest, InvalidType) {
  sql("create table test_table (a integer, b float);");
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN a TYPE invalid_type NOT NULL;",
      "Type definition for column a is invalid: `invalid_type`");
}

TEST_F(AlterTableAlterColumnTest, ScalarTypes) {
  // clang-format off
  createTextTable({"b", "t", "s", "i", "bi", "f", "dc", "tm", "tp", "dt", "dict_text"},
                  {{"True", "100", "30000", "2000000000", "9000000000000000000",
                    "10.1", "100.1234", "00:00:10", "1/1/2000 00:00:59", "1/1/2000",
                    "text_1"},
                   {"False", "110", "30500", "2000500000", "9000000050000000000",
                    "100.12", "2.1234", "00:10:00", "6/15/2020 00:59:59", "6/15/2020",
                    "text_2"},
                   {"True", "120", "31000", "2100000000", "9100000000000000000",
                    "1000.123", "100.1", "10:00:00", "12/31/2500 23:59:59",
                    "12/31/2500", "text_3"},
                   {"NULL", "NULL", "NULL", "NULL", "NULL",
                    "NULL", "NULL", "NULL", "NULL",
                    "NULL", "NULL"}} );
  // clang-format on
  std::string alter_column_command =
      "ALTER TABLE test_table"
      " ALTER COLUMN b TYPE BOOLEAN"
      ", ALTER COLUMN t TYPE TINYINT"
      ", ALTER COLUMN s TYPE SMALLINT"
      ", ALTER COLUMN i TYPE INT"
      ", ALTER COLUMN bi TYPE BIGINT"
      ", ALTER COLUMN f TYPE FLOAT"
      ", ALTER COLUMN dc TYPE DECIMAL(10,5)"
      ", ALTER COLUMN tm TYPE TIME"
      ", ALTER COLUMN tp TYPE TIMESTAMP"
      ", ALTER COLUMN dt TYPE DATE"
      ", ALTER COLUMN dict_text TYPE TEXT ENCODING DICT(32);";
  sql(alter_column_command);
  // clang-format off
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, True, 100L, 30000L, 2000000000L, 9000000000000000000L, 10.1f, 100.1234,
       "00:00:10", "1/1/2000 00:00:59", "1/1/2000", "text_1"},
      {2L, False, 110L, 30500L, 2000500000L, 9000000050000000000L, 100.12f,
       2.1234, "00:10:00", "6/15/2020 00:59:59", "6/15/2020", "text_2"},
      {3L, True, 120L, 31000L, 2100000000L, 9100000000000000000L, 1000.123f,
       100.1, "10:00:00", "12/31/2500 23:59:59", "12/31/2500", "text_3"},
      {4L, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null}
  };
  // clang-format on
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"b", "BOOLEAN"},
      {"t", "TINYINT"},
      {"s", "SMALLINT"},
      {"i", "INT"},
      {"bi", "BIGINT"},
      {"f", "FLOAT"},
      {"dc", "DECIMAL(10,5)"},
      {"tm", "TIME"},
      {"tp", "TIMESTAMP"},
      {"dt", "DATE"},
      {"dict_text", "TEXT ENCODING DICT(32)"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, ArrayTypes) {
  // clang-format off
  createTextTable(
      {"b", "t", "s", "i", "bi", "f", "tm", "tp", "dt", "dict_text",
       "fixedpoint"},
      {{"{True}", "{50, 100}", "{30000, 20000}", "{2000000000}",
        "{9000000000000000000}", "{10.1, 11.1}", "{\"00:00:10\"}",
        "{\"1/1/2000 00:00:59\", \"1/1/2010 00:00:59\"}",
        "{\"1/1/2000\", \"2/2/2000\"}", "{\"text_1\"}", "{1.23,2.34}"},
       {"{False, True}", "{110}", "{30500}", "{2000500000}",
        "{9000000050000000000}", "{100.12}", "{\"00:10:00\", \"00:20:00\"}",
        "{\"6/15/2020 00:59:59\"}", "{\"6/15/2020\"}",
        "{\"text_2\", \"text_3\"}", "{3.456,4.5,5.6}"},
       {"{True}", "{120}", "{31000}", "{2100000000, 200000000}",
        "{9100000000000000000, 9200000000000000000}", "{1000.123}",
        "{\"10:00:00\"}", "{\"12/31/2500 23:59:59\"}", "{\"12/31/2500\"}",
        "{\"text_4\"}", "{6.78}"},
       {"NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL",
        "NULL", "NULL"}});
  // clang-format on
  std::string alter_column_command =
      "ALTER TABLE test_table"
      " ALTER COLUMN b TYPE BOOLEAN[]"
      ", ALTER COLUMN t TYPE TINYINT[]"
      ", ALTER COLUMN s TYPE SMALLINT[]"
      ", ALTER COLUMN i TYPE INT[]"
      ", ALTER COLUMN bi TYPE BIGINT[]"
      ", ALTER COLUMN f TYPE FLOAT[]"
      ", ALTER COLUMN tm TYPE TIME[]"
      ", ALTER COLUMN tp TYPE TIMESTAMP[]"
      ", ALTER COLUMN dt TYPE DATE[]"
      ", ALTER COLUMN dict_text TYPE TEXT[]"
      ", ALTER COLUMN fixedpoint TYPE DECIMAL(10,5)[];";
  sql(alter_column_command);
  // clang-format off
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, array({True}), array({50L, 100L}), array({30000L, 20000L}),
       array({2000000000L}), array({9000000000000000000L}),
       array({10.1f, 11.1f}), array({"00:00:10"}),
       array({"1/1/2000 00:00:59", "1/1/2010 00:00:59"}),
       array({"1/1/2000", "2/2/2000"}), array({"text_1"}), array({1.23, 2.34})},
      {2L, array({False, True}), array({110L}), array({30500L}),
       array({2000500000L}), array({9000000050000000000L}), array({100.12f}),
       array({"00:10:00", "00:20:00"}), array({"6/15/2020 00:59:59"}),
       array({"6/15/2020"}), array({"text_2", "text_3"}),
       array({3.456, 4.5, 5.6})},
      {3L, array({True}), array({120L}), array({31000L}),
       array({2100000000L, 200000000L}),
       array({9100000000000000000L, 9200000000000000000L}), array({1000.123f}),
       array({"10:00:00"}), array({"12/31/2500 23:59:59"}),
       array({"12/31/2500"}), array({"text_4"}), array({6.78})},
      {4L, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null}};
  // clang-format on
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"b", "BOOLEAN[]"},
      {"t", "TINYINT[]"},
      {"s", "SMALLINT[]"},
      {"i", "INT[]"},
      {"bi", "BIGINT[]"},
      {"f", "FLOAT[]"},
      {"tm", "TIME[]"},
      {"tp", "TIMESTAMP[]"},
      {"dt", "DATE[]"},
      {"dict_text", "TEXT[]"},
      {"fixedpoint", "DECIMAL(10,5)[]"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, FixedLengthArrayTypes) {
  // clang-format off
  createTextTable(
      {"b", "t", "s", "i", "bi", "f", "tm", "tp", "dt", "dict_text",
       "fixedpoint"},
      {{"{True,False}", "{50, 100}", "{30000, 20000}", "{2000000000,-100000}",
        "{9000000000000000000,-9000000000000000000}", "{10.1, 11.1}",
        "{\"00:00:10\",\"01:00:10\"}",
        "{\"1/1/2000 00:00:59\", \"1/1/2010 00:00:59\"}",
        "{\"1/1/2000\", \"2/2/2000\"}", "{\"text_1\",\"text_2\"}",
        "{1.23,2.34}"},
       {"{False, True}", "{110,101}", "{30500,10001}", "{2000500000,-23233}",
        "{9000000050000000000,-9200000000000000000}", "{100.12,2.22}",
        "{\"00:10:00\", \"00:20:00\"}",
        "{\"6/15/2020 00:59:59\",\"8/22/2020 00:00:59\"}",
        "{\"6/15/2020\",\"8/22/2020\"}", "{\"text_3\", \"text_4\"}",
        "{3.456,4.5}"},
       {"{True,True}", "{120,44}", "{31000,8123}", "{2100000000, 200000000}",
        "{9100000000000000000, 9200000000000000000}", "{1000.123,1392.22}",
        "{\"10:00:00\",\"20:00:00\"}",
        "{\"12/31/2500 23:59:59\",\"1/1/2500 23:59:59\"}",
        "{\"12/31/2500\",\"1/1/2500\"}", "{\"text_5\",\"text_6\"}",
        "{6.78,5.6}"},
       {"NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL",
        "NULL", "NULL"}});
  // clang-format on
  std::string alter_column_command =
      "ALTER TABLE test_table"
      " ALTER COLUMN b TYPE BOOLEAN[2]"
      ", ALTER COLUMN t TYPE TINYINT[2]"
      ", ALTER COLUMN s TYPE SMALLINT[2]"
      ", ALTER COLUMN i TYPE INT[2]"
      ", ALTER COLUMN bi TYPE BIGINT[2]"
      ", ALTER COLUMN f TYPE FLOAT[2]"
      ", ALTER COLUMN tm TYPE TIME[2]"
      ", ALTER COLUMN tp TYPE TIMESTAMP[2]"
      ", ALTER COLUMN dt TYPE DATE[2]"
      ", ALTER COLUMN dict_text TYPE TEXT[2]"
      ", ALTER COLUMN fixedpoint TYPE DECIMAL(10,5)[2];";
  sql(alter_column_command);
  // clang-format off
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, array({True, False}), array({50L, 100L}), array({30000L, 20000L}),
       array({2000000000L, -100000L}),
       array({9000000000000000000L, -9000000000000000000L}),
       array({10.1f, 11.1f}), array({"00:00:10", "01:00:10"}),
       array({"1/1/2000 00:00:59", "1/1/2010 00:00:59"}),
       array({"1/1/2000", "2/2/2000"}), array({"text_1", "text_2"}),
       array({1.23, 2.34})},
      {2L, array({False, True}), array({110L, 101L}), array({30500L, 10001L}),
       array({2000500000L, -23233L}),
       array({9000000050000000000L, -9200000000000000000L}),
       array({100.12f, 2.22f}), array({"00:10:00", "00:20:00"}),
       array({"6/15/2020 00:59:59", "8/22/2020 00:00:59"}),
       array({"6/15/2020", "8/22/2020"}), array({"text_3", "text_4"}),
       array({3.456, 4.5})},
      {3L, array({True, True}), array({120L, 44L}), array({31000L, 8123L}),
       array({2100000000L, 200000000L}),
       array({9100000000000000000L, 9200000000000000000L}),
       array({1000.123f, 1392.22f}), array({"10:00:00", "20:00:00"}),
       array({"12/31/2500 23:59:59", "1/1/2500 23:59:59"}),
       array({"12/31/2500", "1/1/2500"}), array({"text_5", "text_6"}),
       array({6.78, 5.6})},
      {4L, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null}};
  // clang-format on
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"b", "BOOLEAN[2]"},
      {"t", "TINYINT[2]"},
      {"s", "SMALLINT[2]"},
      {"i", "INT[2]"},
      {"bi", "BIGINT[2]"},
      {"f", "FLOAT[2]"},
      {"tm", "TIME[2]"},
      {"tp", "TIMESTAMP[2]"},
      {"dt", "DATE[2]"},
      {"dict_text", "TEXT[2]"},
      {"fixedpoint", "DECIMAL(10,5)[2]"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, GeoTypes) {
  // clang-format off
  createTextTable(
      {"p", "mpoint", "l", "mlinestring", "poly", "multipoly"},
      {{"POINT (0 0)", "MULTIPOINT (0 0,1 1)", "LINESTRING (0 0,0 0)",
        "MULTILINESTRING ((0 0,1 1),(2 2,3 3))", "POLYGON ((0 0,1 0,1 1,0 1,0 0))", 
        "MULTIPOLYGON (((0 0,1 0,0 1,0 0)))"},
       {"NULL", "NULL", "NULL", "NULL", "NULL", "NULL"},
       {"POINT (1 1)", "MULTIPOINT (1 1,2 2)", "LINESTRING (1 1,2 2,3 3)",
        "MULTILINESTRING ((1 1,2 2),(3 3,4 4))", "POLYGON ((5 4,7 4,6 5,5 4))",
        "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"},
       {"POINT (2 2)", "MULTIPOINT (3 4,4 3,0 0)", "LINESTRING (2 2,3 3)", "MULTILINESTRING ((2 2,3 3),(4 4,5 5))", "POLYGON ((1 1,3 1,2 3,1 1))",
        "MULTIPOLYGON (((0 0,3 0,0 3,0 0)),((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 "
        "0)))"},
       {"NULL", "NULL", "NULL", "NULL", "NULL", "NULL"}});
  // clang-format on
  std::string alter_column_command =
      "ALTER TABLE test_table"
      " ALTER COLUMN p TYPE POINT"
      ", ALTER COLUMN mpoint TYPE MULTIPOINT"
      ", ALTER COLUMN l TYPE LINESTRING"
      ", ALTER COLUMN mlinestring TYPE MULTILINESTRING"
      ", ALTER COLUMN poly TYPE POLYGON"
      ", ALTER COLUMN multipoly TYPE MULTIPOLYGON;";
  sql(alter_column_command);
  // clang-format off
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {i(1), "POINT (0 0)", "MULTIPOINT (0 0,1 1)", "LINESTRING (0 0,0 0)", "MULTILINESTRING ((0 0,1 1),(2 2,3 3))",
       "POLYGON ((0 0,1 0,1 1,0 1,0 0))", "MULTIPOLYGON (((0 0,1 0,0 1,0 0)))"},
      {i(2), Null, Null, Null, Null, Null, Null},
      {i(3), "POINT (1 1)", "MULTIPOINT (1 1,2 2)", "LINESTRING (1 1,2 2,3 3)", "MULTILINESTRING ((1 1,2 2),(3 3,4 4))",
       "POLYGON ((5 4,7 4,6 5,5 4))",
       "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"},
      {i(4), "POINT (2 2)", "MULTIPOINT (3 4,4 3,0 0)", "LINESTRING (2 2,3 3)", "MULTILINESTRING ((2 2,3 3),(4 4,5 5))",
       "POLYGON ((1 1,3 1,2 3,1 1))",
       "MULTIPOLYGON (((0 0,3 0,0 3,0 0)),((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 "
       "0)))"},
      {i(5), Null, Null, Null, Null, Null, Null}};
  // clang-format on
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"p", "POINT"},
      {"mpoint", "MULTIPOINT"},
      {"l", "LINESTRING"},
      {"mlinestring", "MULTILINESTRING"},
      {"poly", "POLYGON"},
      {"multipoly", "MULTIPOLYGON"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, ReencodeDictionaryLowerDepth) {
  createTextTable({"d0", "d1"},
                  {
                      {"text 11", "text 21"},
                      {"text 12", "text 22"},
                      {"text 13", "text 23"},
                  },
                  "ENCODING DICT (32)");

  std::string alter_column_command =
      "ALTER TABLE test_table"
      " ALTER COLUMN d0 TYPE TEXT ENCODING DICT(16)"
      ", ALTER COLUMN d1 TYPE TEXT ENCODING DICT(8);";
  sql(alter_column_command);
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{{
      {1L, "text 11", "text 21"},
      {2L, "text 12", "text 22"},
      {3L, "text 13", "text 23"},
  }};
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"d0", "TEXT ENCODING DICT(16)"},
      {"d1", "TEXT ENCODING DICT(8)"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, ReencodeDictionaryHigherDepth) {
  createTextTable({"d0", "d1"},
                  {
                      {"text 11", "text 21"},
                      {"text 12", "text 22"},
                      {"text 13", "text 23"},
                  },
                  "ENCODING DICT (8)");

  std::string alter_column_command =
      "ALTER TABLE test_table"
      " ALTER COLUMN d0 TYPE TEXT ENCODING DICT(16)"
      ", ALTER COLUMN d1 TYPE TEXT ENCODING DICT(32);";
  sql(alter_column_command);
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{{
      {1L, "text 11", "text 21"},
      {2L, "text 12", "text 22"},
      {3L, "text 13", "text 23"},
  }};
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"d0", "TEXT ENCODING DICT(16)"},
      {"d1", "TEXT ENCODING DICT(32)"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, DictEncodedToNoneEncodedString) {
  createTextTable({"txt"}, {{"txt1"}, {"txt2"}, {"txt3"}}, "ENCODING DICT(32)");
  sql("ALTER TABLE test_table ALTER COLUMN txt TYPE TEXT ENCODING NONE;");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, "txt1"},
      {2L, "txt2"},
      {3L, "txt3"},
  };
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"txt", "TEXT ENCODING NONE"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, MultiFragment) {
  createTextTable({"bi", "i"},
                  {{"100", "100"}, {"200", "200"}, {"300", "300"}},
                  "ENCODING DICT(32)",
                  "test_table",
                  false,
                  "fragment_size=1");
  sql("ALTER TABLE test_table ALTER COLUMN i TYPE INT, ALTER COLUMN bi TYPE BIGINT;");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, 100L, 100L},
      {2L, 200L, 200L},
      {3L, 300L, 300L},
  };
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"i", "INT"},
      {"bi", "BIGINT"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, MultiFragmentMalformed) {
  createTextTable({"bi", "i"},
                  {{"12", "123"}, {"1234", "12345"}, {"123456", "3000000000000"}},
                  "ENCODING DICT(32)",
                  "test_table",
                  false,
                  "fragment_size=1");
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN i TYPE INT, ALTER COLUMN bi TYPE BIGINT;",
      "Alter column type: Column i: Integer 3000000000000 is out of range for INTEGER");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, "12", "123"},
      {2L, "1234", "12345"},
      {3L, "123456", "3000000000000"},
  };
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"i", "TEXT"},
      {"bi", "TEXT"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, MultiFragmentGeoTypes) {
  createTextTable({"p", "l"},
                  {{"POINT (0 0)", "LINESTRING (1 1,2 2,3 3)"},
                   {"POINT (2 2)", "LINESTRING (2 2,3 3)"},
                   {"POINT (3 3)", "LINESTRING (1 1,3 3)"}},
                  "ENCODING DICT(32)",
                  "test_table",
                  false,
                  "fragment_size=1");
  sql("ALTER TABLE test_table ALTER COLUMN p TYPE POINT, ALTER COLUMN l TYPE "
      "LINESTRING;");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, "POINT (0 0)", "LINESTRING (1 1,2 2,3 3)"},
      {2L, "POINT (2 2)", "LINESTRING (2 2,3 3)"},
      {3L, "POINT (3 3)", "LINESTRING (1 1,3 3)"}};
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"p", "POINT"},
      {"l", "LINESTRING"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, MultiFragmentMalformedGeoTypes) {
  createTextTable({"p", "l"},
                  {{"POINT (0 0)", "LINESTRING (1 1,2 2,3 3)"},
                   {"POINT (2 2)", "LINESTRING (2 2,3 3)"},
                   {"POINT (3 3)", "LINESTRING xx (1 1,3 3)"}},
                  "ENCODING DICT(32)",
                  "test_table",
                  false,
                  "fragment_size=1");
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN p TYPE POINT, ALTER COLUMN l TYPE LINESTRING;",
      "Alter column type: Column l: Failed to extract valid geometry in HeavyDB column "
      "'l'.");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, "POINT (0 0)", "LINESTRING (1 1,2 2,3 3)"},
      {2L, "POINT (2 2)", "LINESTRING (2 2,3 3)"},
      {3L, "POINT (3 3)", "LINESTRING xx (1 1,3 3)"}};
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"p", "TEXT"},
      {"l", "TEXT"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, NonString) {
  sql("CREATE TABLE test_table (index INT);");
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN index TYPE BIGINT;",
      "Altering column index type not allowed. Column type must be TEXT.");
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, ShardedTable) {
  sql("CREATE TABLE test_table (index TEXT, "
      "SHARD KEY (index)) WITH (shard_count=2);");
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN index TYPE BIGINT;",
      "Alter column type: Column index: altering a sharded table is unsupported");
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "TEXT"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_F(AlterTableAlterColumnTest, EmptyTable) {
  sql("CREATE TABLE test_table (index INT, txt TEXT ENCODING NONE);");
  sql("ALTER TABLE test_table ALTER COLUMN txt TYPE INT;");
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", {});
  {
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"txt", "INT"},
    };
    compareSchemaToReference("test_table", reference_schema);
  }
  sql("INSERT INTO test_table VALUES (1,100), (2,101), (3,102);");
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;",
                      {{1L, 100L}, {2L, 101L}, {3L, 102L}});
  {
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"txt", "INT"},
    };
    compareSchemaToReference("test_table", reference_schema);
  }
}

TEST_F(AlterTableAlterColumnTest, NullTextValuesAlterToNotNull) {
  createTextTable(
      {"txt"}, {{"txt1"}, {"NULL"}, {"NULL"}}, "ENCODING DICT(32)", "test_table", true);
  sql("ALTER TABLE test_table ALTER COLUMN txt TYPE TEXT NOT NULL;");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, "txt1"},
      {2L, "NULL"},
      {3L, "NULL"},
  };
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  compareSchemaToReference("test_table",
                           {
                               {"index", "INT"},
                               {"txt", "TEXT NOT NULL"},
                           });
}

TEST_F(AlterTableAlterColumnTest, NullValuesAlterToNotNull) {
  createTextTable({"txt"}, {{"txt1"}, {"NULL"}, {"NULL"}});
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN txt TYPE TEXT NOT NULL;",
      "Alter column type: Column txt: NULL value not allowed in NOT NULL column");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, "txt1"},
      {2L, Null},
      {3L, Null},
  };
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
}

TEST_F(AlterTableAlterColumnTest, DefaultValues) {
  sql("CREATE TABLE test_table (index INT, txt TEXT DEFAULT 'a' ENCODING NONE);");
  sql("INSERT INTO test_table (index) VALUES (1), (2), (3);");
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;",
                      {
                          {1L, "a"},
                          {2L, "a"},
                          {3L, "a"},
                      });
  compareSchemaToReference("test_table",
                           {
                               {"index", "INT"},
                               {"txt", "TEXT DEFAULT 'a' ENCODING NONE"},
                           });
  sql("ALTER TABLE test_table ALTER COLUMN txt TYPE TEXT DEFAULT 'b';");
  sql("INSERT INTO test_table (index) VALUES (4), (5), (6);");
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;",
                      {
                          {1L, "a"},
                          {2L, "a"},
                          {3L, "a"},
                          {4L, "b"},
                          {5L, "b"},
                          {6L, "b"},
                      });
  compareSchemaToReference("test_table",
                           {
                               {"index", "INT"},
                               {"txt", "TEXT DEFAULT 'b'"},
                           });
  sql("ALTER TABLE test_table ALTER COLUMN txt TYPE TEXT DEFAULT NULL;");
  sql("INSERT INTO test_table (index) VALUES (7), (8), (9);");
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;",
                      {
                          {1L, "a"},
                          {2L, "a"},
                          {3L, "a"},
                          {4L, "b"},
                          {5L, "b"},
                          {6L, "b"},
                          {7L, Null},
                          {8L, Null},
                          {9L, Null},
                      });
  compareSchemaToReference("test_table",
                           {
                               {"index", "INT"},
                               {"txt", "TEXT DEFAULT NULL"},
                           });
}

TEST_F(AlterTableAlterColumnTest, NonExistentTable) {
  queryAndAssertException(
      "ALTER TABLE non_existent_table ALTER COLUMN txt TYPE TEXT;",
      "Table/View non_existent_table for catalog heavyai does not exist.");
}

TEST_F(AlterTableAlterColumnTest, NonExistentColumn) {
  sql("CREATE TABLE test_table (index INT, txt TEXT ENCODING NONE);");
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN txt_non_existent TYPE TEXT;",
      "Column txt_non_existent does not exist.");
}

TEST_F(AlterTableAlterColumnTest, View) {
  sql("CREATE TABLE test_table (index INT, txt TEXT ENCODING NONE);");
  sql("INSERT INTO test_table (index) VALUES (1), (2), (3);");
  sql("CREATE VIEW test_view AS SELECT * FROM test_table;");
  queryAndAssertException("ALTER TABLE test_view ALTER COLUMN txt TYPE TEXT;",
                          "Altering columns in a view is not supported.");
}

TEST_F(AlterTableAlterColumnTest, TemporaryTable) {
  sql("CREATE TEMPORARY TABLE test_temp_table (index INT, txt TEXT ENCODING NONE);");
  sql("INSERT INTO test_temp_table (index) VALUES (1), (2), (3);");
  queryAndAssertException("ALTER TABLE test_temp_table ALTER COLUMN txt TYPE TEXT;",
                          "Altering columns in temporary tables is not supported.");
}

TEST_F(AlterTableAlterColumnTest, ForeignTable) {
  sql("CREATE FOREIGN TABLE test_foreign_table (a INT) SERVER default_local_delimited "
      "WITH (file_path='../../Tests/FsiDataFiles/1.csv');");
  queryAndAssertException(
      "ALTER TABLE test_foreign_table ALTER COLUMN txt TYPE TEXT;",
      "test_foreign_table is a foreign table. Use ALTER FOREIGN TABLE.");
}

TEST_F(AlterTableAlterColumnTest, UserWithoutPermissions) {
  createTestUser();
  sql("CREATE TABLE test_table (index INT, txt TEXT ENCODING NONE);");
  sql("INSERT INTO test_table (index) VALUES (1), (2), (3);");
  login("test_user", "test_pass");
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN txt TYPE TEXT;",
      "Current user does not have the privilege to alter table: test_table");
  loginAdmin();
  sql("DROP USER test_user;");
}

class AlterColumnScalarTypesOutOfRange : public AlterTableAlterColumnTest {
 protected:
  static void SetUpTestSuite() {
    sql("DROP TABLE IF EXISTS test_suite_table;");
    // clang-format off
    createTextTable( {"ti"},
                  {
                      {"127"},
                      {"1200"},
                      {"18000000001"},
                  },
                    "ENCODING NONE",
                    "test_suite_table");
    // clang-format on
  }

  static void TearDownTestSuite() { sql("DROP TABLE IF EXISTS test_suite_table;"); }

  void assertDataAndSchemaNotChanged() {
    // clang-format off
    auto expected_values = std::vector<std::vector<NullableTargetValue>>{
                          {1L, "127"},
                          {2L, "1200"},
                          {3L, "18000000001"},
                      };
    // clang-format on
    sqlAndCompareResult("SELECT * FROM test_suite_table ORDER BY index;",
                        expected_values);
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"ti", "TEXT ENCODING NONE"},
    };
    compareSchemaToReference("test_suite_table", reference_schema);
  }
};

TEST_F(AlterColumnScalarTypesOutOfRange, TinyInt) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN ti TYPE TINYINT;",
      "Alter column type: Column ti: Integer 1200 is out of range for TINYINT");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnScalarTypesOutOfRange, BigInt8BitFixedEncoding) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN ti TYPE BIGINT ENCODING FIXED (8);",
      "Alter column type: Column ti: Integer 1200 exceeds maximum value for BIGINT(8)");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnScalarTypesOutOfRange, Int) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN ti TYPE INT;",
      "Alter column type: Column ti: Integer 18000000001 is out of range for INTEGER");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnScalarTypesOutOfRange, BigInt32BitFixedEncoding) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN ti TYPE BIGINT ENCODING FIXED (32);",
      "Alter column type: Column ti: Integer 18000000001 exceeds maximum value for "
      "BIGINT(32)");
  assertDataAndSchemaNotChanged();
}

class AlterColumnArrayTypesOutOfRange : public AlterTableAlterColumnTest {
 protected:
  static void SetUpTestSuite() {
    sql("DROP TABLE IF EXISTS test_suite_table;");
    // clang-format off
    createTextTable( {"ti"}, {{"{127,127}"}, {"{1200}"}, {"{18000000001}"}},
        "ENCODING NONE",
                    "test_suite_table");
    // clang-format on
  }

  static void TearDownTestSuite() { sql("DROP TABLE IF EXISTS test_suite_table;"); }

  void assertDataAndSchemaNotChanged() {
    // clang-format off
    auto expected_values = std::vector<std::vector<NullableTargetValue>>
{{1L, "{127,127}"}, {2L, "{1200}"}, {3L, "{18000000001}"}};
    // clang-format on
    sqlAndCompareResult("SELECT * FROM test_suite_table ORDER BY index;",
                        expected_values);
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"ti", "TEXT ENCODING NONE"},
    };
    compareSchemaToReference("test_suite_table", reference_schema);
  }
};

TEST_F(AlterColumnArrayTypesOutOfRange, TinyInt) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN ti TYPE TINYINT[];",
      "Alter column type: Column ti: Integer 1200 is out of range for TINYINT");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnArrayTypesOutOfRange, Int) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN ti TYPE INT[];",
      "Alter column type: Column ti: Integer 18000000001 is out of range for INTEGER");
  assertDataAndSchemaNotChanged();
}

class AlterColumnFixedArrayTypesOutOfRange : public AlterTableAlterColumnTest {
 protected:
  static void SetUpTestSuite() {
    sql("DROP TABLE IF EXISTS test_suite_table;");
    // clang-format off
    createTextTable( {"ti"}, {{"{127,126}"}, {"{1200,1201}"}, {"{18000000001,18000000002}"}},
        "ENCODING NONE",
                    "test_suite_table");
    // clang-format on
  }

  static void TearDownTestSuite() { sql("DROP TABLE IF EXISTS test_suite_table;"); }

  void assertDataAndSchemaNotChanged() {
    // clang-format off
    auto expected_values = std::vector<std::vector<NullableTargetValue>>
{{1L, "{127,126}"}, {2L, "{1200,1201}"}, {3L, "{18000000001,18000000002}"}};
    // clang-format on
    sqlAndCompareResult("SELECT * FROM test_suite_table ORDER BY index;",
                        expected_values);
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"ti", "TEXT ENCODING NONE"},
    };
    compareSchemaToReference("test_suite_table", reference_schema);
  }
};

TEST_F(AlterColumnFixedArrayTypesOutOfRange, TinyInt) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN ti TYPE TINYINT[2];",
      "Alter column type: Column ti: Integer 1200 is out of range for TINYINT");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnFixedArrayTypesOutOfRange, Int) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN ti TYPE INT[2];",
      "Alter column type: Column ti: Integer 18000000001 is out of range for INTEGER");
  assertDataAndSchemaNotChanged();
}

class AlterColumnMalformedGeoTest : public AlterTableAlterColumnTest {
 protected:
  static void SetUpTestSuite() {
    sql("DROP TABLE IF EXISTS test_suite_table;");
    createTextTable({"p", "l", "poly", "multipoly"},
                    {{"POINT (x 0)",
                      "LINESTRING (x 0,0 0)",
                      "POLYGON ((x 0,1 0,1 1,0 1,0 0))",
                      "MULTIPOLYGON (((x 0,1 0,0 1,0 0)))"}},
                    "ENCODING NONE",
                    "test_suite_table");
  }

  static void TearDownTestSuite() { sql("DROP TABLE IF EXISTS test_suite_table;"); }

  void assertDataAndSchemaNotChanged() {
    auto expected_values = std::vector<std::vector<NullableTargetValue>>{
        {1L,
         "POINT (x 0)",
         "LINESTRING (x 0,0 0)",
         "POLYGON ((x 0,1 0,1 1,0 1,0 0))",
         "MULTIPOLYGON (((x 0,1 0,0 1,0 0)))"}};
    sqlAndCompareResult("SELECT * FROM test_suite_table ORDER BY index;",
                        expected_values);
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"p", "TEXT ENCODING NONE"},
        {"l", "TEXT ENCODING NONE"},
        {"poly", "TEXT ENCODING NONE"},
        {"multipoly", "TEXT ENCODING NONE"},
    };
    compareSchemaToReference("test_suite_table", reference_schema);
  }
};

TEST_F(AlterColumnMalformedGeoTest, Point) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN p TYPE POINT;",
                          "Alter column type: Column p: Failed to extract valid geometry "
                          "in HeavyDB column 'p'.");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedGeoTest, Linestring) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN l TYPE LINESTRING;",
                          "Alter column type: Column l: Failed to extract valid geometry "
                          "in HeavyDB column 'l'.");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedGeoTest, Polygon) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN poly TYPE POLYGON;",
                          "Alter column type: Column poly: Failed to extract valid "
                          "geometry in HeavyDB column 'poly'.");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedGeoTest, MultiPolygon) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN multipoly TYPE MULTIPOLYGON;",
      "Alter column type: Column multipoly: Failed to extract valid geometry in HeavyDB "
      "column 'multipoly'.");
  assertDataAndSchemaNotChanged();
}

class AlterColumnMalformedScalarTest : public AlterTableAlterColumnTest {
 protected:
  static void SetUpTestSuite() {
    sql("DROP TABLE IF EXISTS test_suite_table;");
    // clang-format off
    createTextTable({"b", "t", "s", "i", "bi", "f", "dc", "tm", "tp", "dt"},
                    {
                        {"Truex", "100x", "30000x", "2000000000x",
                         "9000000000000000000x", "10.1x", "100.1234x",
                         "00:00:10x", "xx1x/1/2000 00:00:59x", "xx1/1/2000x"},
                    },
                    "ENCODING NONE", "test_suite_table");
    // clang-format on
  }

  static void TearDownTestSuite() { sql("DROP TABLE IF EXISTS test_suite_table;"); }

  void assertDataAndSchemaNotChanged() {
    // clang-format off
    auto expected_values = std::vector<std::vector<NullableTargetValue>>{
        {1L, "Truex", "100x", "30000x", "2000000000x", "9000000000000000000x",
         "10.1x", "100.1234x", "00:00:10x", "xx1x/1/2000 00:00:59x",
         "xx1/1/2000x"}};
    // clang-format on

    sqlAndCompareResult("SELECT * FROM test_suite_table ORDER BY index;",
                        expected_values);
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"b", "TEXT ENCODING NONE"},
        {"t", "TEXT ENCODING NONE"},
        {"s", "TEXT ENCODING NONE"},
        {"i", "TEXT ENCODING NONE"},
        {"bi", "TEXT ENCODING NONE"},
        {"f", "TEXT ENCODING NONE"},
        {"dc", "TEXT ENCODING NONE"},
        {"tm", "TEXT ENCODING NONE"},
        {"tp", "TEXT ENCODING NONE"},
        {"dt", "TEXT ENCODING NONE"},
    };
    compareSchemaToReference("test_suite_table", reference_schema);
  }
};

TEST_F(AlterColumnMalformedScalarTest, Boolean) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN b TYPE BOOLEAN;",
      "Alter column type: Column b: Invalid string for boolean Truex");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedScalarTest, TinyInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN t TYPE TINYINT;",
                          "Alter column type: Column t: Unexpected character \"x\" "
                          "encountered in TINYINT value 100x");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedScalarTest, SmallInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN s TYPE SMALLINT;",
                          "Alter column type: Column s: Unexpected character \"x\" "
                          "encountered in SMALLINT value 30000x");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedScalarTest, Int) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN i TYPE INT;",
                          "Alter column type: Column i: Unexpected character \"x\" "
                          "encountered in INTEGER value 2000000000x");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedScalarTest, BigInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN bi TYPE BIGINT;",
                          "Alter column type: Column bi: Unexpected character \"x\" "
                          "encountered in BIGINT value 9000000000000000000x");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedScalarTest, Time) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN tm TYPE TIME;",
      "Alter column type: Column tm: Invalid TIME string (00:00:10x)");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedScalarTest, Timestamp) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN tp TYPE TIMESTAMP;",
      "Alter column type: Column tp: Invalid TIMESTAMP string (xx1x/1/2000 00:00:59x)");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedScalarTest, Date) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN dt TYPE DATE;",
      "Alter column type: Column dt: Invalid DATE string (xx1/1/2000x)");
  assertDataAndSchemaNotChanged();
}

// TODO: both float and decimal appear to incorreclty ignore malformed input, this will
// be addressed as a separate work item, and tests will need to be added

class AlterColumnMalformedArrayTest : public AlterTableAlterColumnTest {
 protected:
  static void SetUpTestSuite() {
    sql("DROP TABLE IF EXISTS test_suite_table;");
    // clang-format off
    createTextTable({"b", "t", "s", "i", "bi", "f", "tm", "tp", "dt", "fixedpoint"},
                    {{"{Truex}", "{50x, 100x}", "{30000x, 20000x}", "{2000000000x}",
                      "{9000000000000000000x}", "{10.1x, 11.1x}",
                      "{\"xx00:00:10x\"}",
                      "{\"xx1/1/2000 x00:00:59\", \"1/1x/2010 00:00:59\"}",
                      "{\"1/x1/2000xx\", \"2/2/2000\"}", "{1.23xx,2.34}"}},
                    "ENCODING NONE",
                    "test_suite_table");
    // clang-format on
  }

  static void TearDownTestSuite() { sql("DROP TABLE IF EXISTS test_suite_table;"); }

  void assertDataAndSchemaNotChanged() {
    // clang-format off
    auto expected_values = std::vector<std::vector<NullableTargetValue>>{
        {1L, "{Truex}", "{50x, 100x}", "{30000x, 20000x}", "{2000000000x}",
         "{9000000000000000000x}", "{10.1x, 11.1x}", "{\"xx00:00:10x\"}",
         "{\"xx1/1/2000 x00:00:59\", \"1/1x/2010 00:00:59\"}",
         "{\"1/x1/2000xx\", \"2/2/2000\"}", "{1.23xx,2.34}"}};
    // clang-format on
    sqlAndCompareResult("SELECT * FROM test_suite_table ORDER BY index;",
                        expected_values);
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"b", "TEXT ENCODING NONE"},
        {"t", "TEXT ENCODING NONE"},
        {"s", "TEXT ENCODING NONE"},
        {"i", "TEXT ENCODING NONE"},
        {"bi", "TEXT ENCODING NONE"},
        {"f", "TEXT ENCODING NONE"},
        {"tm", "TEXT ENCODING NONE"},
        {"tp", "TEXT ENCODING NONE"},
        {"dt", "TEXT ENCODING NONE"},
        {"fixedpoint", "TEXT ENCODING NONE"},
    };
    compareSchemaToReference("test_suite_table", reference_schema);
  }
};

TEST_F(AlterColumnMalformedArrayTest, Boolean) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN b TYPE BOOLEAN[];",
      "Alter column type: Column b: Invalid string for boolean Truex");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedArrayTest, TinyInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN t TYPE TINYINT[];",
                          "Alter column type: Column t: Unexpected character \"x\" "
                          "encountered in TINYINT value 50x");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedArrayTest, SmallInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN s TYPE SMALLINT[]",
                          "Alter column type: Column s: Unexpected character \"x\" "
                          "encountered in SMALLINT value 30000x");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedArrayTest, Int) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN i TYPE INT[]",
                          "Alter column type: Column i: Unexpected character \"x\" "
                          "encountered in INTEGER value 2000000000x");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedArrayTest, BigInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN bi TYPE BIGINT[]",
                          "Alter column type: Column bi: Unexpected character \"x\" "
                          "encountered in BIGINT value 9000000000000000000x");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedArrayTest, Time) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN tm TYPE TIME[]",
      "Alter column type: Column tm: Invalid TIME string (xx00:00:10x)");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedArrayTest, Timestamp) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN tp TYPE TIMESTAMP[]",
      "Alter column type: Column tp: Invalid TIMESTAMP string (xx1/1/2000 x00:00:59)");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedArrayTest, Date) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN dt TYPE DATE[]",
      "Alter column type: Column dt: Invalid DATE string (1/x1/2000xx)");
  assertDataAndSchemaNotChanged();
}

// TODO: similar to above FLOAT[] and DECIMAL[] do not detect malformed input

class AlterColumnMalformedFixedLengthArrayTest : public AlterTableAlterColumnTest {
 protected:
  static void SetUpTestSuite() {
    sql("DROP TABLE IF EXISTS test_suite_table;");
    // clang-format off
    createTextTable({"b", "t", "s", "i", "bi", "f", "tm", "tp", "dt", "fixedpoint"},
                    {{"{True}", "{50x, 100x}", "{30000x, 20000x}", "{2000000000x}",
                      "{9000000000000000000x}", "{10.1x, 11.1x}", "{\"00:00:10\"}",
                      "{\"xx1/1/2000 x00:00:59\", \"1/1x/2010 00:00:59\"}",
                      "{\"1/x1/2000xx\", \"2/2/2000\"}", "{1.23xx,2.34}"}},
                    "ENCODING NONE",
                    "test_suite_table");
    // clang-format on
  }

  static void TearDownTestSuite() { sql("DROP TABLE IF EXISTS test_suite_table;"); }

  void assertDataAndSchemaNotChanged() {
    // clang-format off
    auto expected_values = std::vector<std::vector<NullableTargetValue>>{
        {1L, "{True}", "{50x, 100x}", "{30000x, 20000x}", "{2000000000x}",
         "{9000000000000000000x}", "{10.1x, 11.1x}", "{\"00:00:10\"}",
         "{\"xx1/1/2000 x00:00:59\", \"1/1x/2010 00:00:59\"}",
         "{\"1/x1/2000xx\", \"2/2/2000\"}", "{1.23xx,2.34}"}};
    // clang-format on
    sqlAndCompareResult("SELECT * FROM test_suite_table ORDER BY index;",
                        expected_values);
    auto reference_schema = std::vector<std::pair<std::string, std::string>>{
        {"index", "INT"},
        {"b", "TEXT ENCODING NONE"},
        {"t", "TEXT ENCODING NONE"},
        {"s", "TEXT ENCODING NONE"},
        {"i", "TEXT ENCODING NONE"},
        {"bi", "TEXT ENCODING NONE"},
        {"f", "TEXT ENCODING NONE"},
        {"tm", "TEXT ENCODING NONE"},
        {"tp", "TEXT ENCODING NONE"},
        {"dt", "TEXT ENCODING NONE"},
        {"fixedpoint", "TEXT ENCODING NONE"},
    };
    compareSchemaToReference("test_suite_table", reference_schema);
  }
};

TEST_F(AlterColumnMalformedFixedLengthArrayTest, Boolean) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN b TYPE BOOLEAN[2];",
                          "Alter column type: Column b: Incorrect number of elements (1) "
                          "in array for fixed length array of size 2");
  assertDataAndSchemaNotChanged();
}

TEST_F(AlterColumnMalformedFixedLengthArrayTest, TinyInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN t TYPE TINYINT[2];",
                          "Alter column type: Column t: Unexpected character \"x\" "
                          "encountered in TINYINT value 50x");
  assertDataAndSchemaNotChanged();
}
TEST_F(AlterColumnMalformedFixedLengthArrayTest, SmallInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN s TYPE SMALLINT[2]",
                          "Alter column type: Column s: Unexpected character \"x\" "
                          "encountered in SMALLINT value 30000x");
  assertDataAndSchemaNotChanged();
}
TEST_F(AlterColumnMalformedFixedLengthArrayTest, Int) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN i TYPE INT[2]",
                          "Alter column type: Column i: Unexpected character \"x\" "
                          "encountered in INTEGER value 2000000000x");
  assertDataAndSchemaNotChanged();
}
TEST_F(AlterColumnMalformedFixedLengthArrayTest, BigInt) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN bi TYPE BIGINT[2]",
                          "Alter column type: Column bi: Unexpected character \"x\" "
                          "encountered in BIGINT value 9000000000000000000x");
  assertDataAndSchemaNotChanged();
}
TEST_F(AlterColumnMalformedFixedLengthArrayTest, Time) {
  queryAndAssertException("ALTER TABLE test_suite_table ALTER COLUMN tm TYPE TIME[2]",
                          "Alter column type: Column tm: Incorrect number of elements "
                          "(1) in array for fixed length array of size 2");
  assertDataAndSchemaNotChanged();
}
TEST_F(AlterColumnMalformedFixedLengthArrayTest, Timestamp) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN tp TYPE TIMESTAMP[2]",
      "Alter column type: Column tp: Invalid TIMESTAMP string (xx1/1/2000 x00:00:59)");
  assertDataAndSchemaNotChanged();
}
TEST_F(AlterColumnMalformedFixedLengthArrayTest, Date) {
  queryAndAssertException(
      "ALTER TABLE test_suite_table ALTER COLUMN dt TYPE DATE[2]",
      "Alter column type: Column dt: Invalid DATE string (1/x1/2000xx)");
  assertDataAndSchemaNotChanged();
}

// TODO: similar to above FLOAT[2] and DECIMAL[2] do not detect malformed input

struct InputSourceAlterColumnTestParam {
  std::string display_type;
  std::string type;
};

class InputSourceAlterColumnTest
    : public AlterTableAlterColumnTest,
      public ::testing::WithParamInterface<InputSourceAlterColumnTestParam> {
 public:
  static std::string testParamsToString(const InputSourceAlterColumnTestParam& param) {
    return param.display_type;
  }
};

INSTANTIATE_TEST_SUITE_P(
    StringInputTypes,
    InputSourceAlterColumnTest,
    ::testing::Values(
        InputSourceAlterColumnTestParam{"text_none_encoded", "ENCODING NONE"},
        InputSourceAlterColumnTestParam{"text_dict_encoded", "ENCODING DICT(32)"}),
    [](const auto& info) {
      return InputSourceAlterColumnTest::testParamsToString(info.param);
    });

TEST_P(InputSourceAlterColumnTest, NoChange) {
  createTextTable({"txt"}, {{"100"}, {"101"}, {"102"}}, GetParam().type);
  queryAndAssertException(
      "ALTER TABLE test_table ALTER COLUMN txt TYPE TEXT " + GetParam().type + ";",
      "Altering column txt results in no change to column, please review command.");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {1L, "100"},
      {2L, "101"},
      {3L, "102"},
  };
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"txt", "TEXT " + GetParam().type},
  };
  compareSchemaToReference("test_table", reference_schema);
}

class AlterColumnMultipleStringSourceTest
    : public AlterTableAlterColumnTest,
      public ::testing::WithParamInterface<std::string> {
 protected:
  std::string getStringEncoding() {
    std::string encoding = "ENCODING NONE";
    if (GetParam() == "dict") {
      encoding = "ENCODING DICT(32)";
    } else {
      CHECK_EQ(GetParam(), "none");
    }
    return encoding;
  }
};

TEST_P(AlterColumnMultipleStringSourceTest, EncodedDates) {
  createTextTable(
      {"date32", "date16"}, {{"6/15/2020", "6/15/2020"}}, getStringEncoding());
  sql("ALTER TABLE test_table ALTER COLUMN date32 TYPE DATE ENCODING FIXED(32), ALTER "
      "COLUMN date16 TYPE DATE ENCODING FIXED(16);");
  auto expected_values =
      std::vector<std::vector<NullableTargetValue>>{{1L, "6/15/2020", "6/15/2020"}};
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"date32", "DATE ENCODING FIXED(32)"},
      {"date16", "DATE ENCODING FIXED(16)"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

TEST_P(AlterColumnMultipleStringSourceTest, EncodedHighPrecisionTimestamps) {
  createTextTable({"ts3", "ts6", "ts9"},
                  {{"12/31/2500 23:59:59.123",
                    "12/31/2500 23:59:59.123456",
                    "12/31/2500 23:59:59.123456789"}},
                  getStringEncoding());
  sql("ALTER TABLE test_table ALTER COLUMN ts3 TYPE TIMESTAMP(3), ALTER COLUMN ts6 TYPE "
      "TIMESTAMP(6), ALTER COLUMN ts9 TYPE TIMESTAMP(9);");
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{{
      1L,
      "12/31/2500 23:59:59.123",
      "12/31/2500 23:59:59.123456",
      "12/31/2500 23:59:59.123456789",
  }};
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY index;", expected_values);
  auto reference_schema = std::vector<std::pair<std::string, std::string>>{
      {"index", "INT"},
      {"ts3", "TIMESTAMP(3)"},
      {"ts6", "TIMESTAMP(6)"},
      {"ts9", "TIMESTAMP(9)"},
  };
  compareSchemaToReference("test_table", reference_schema);
}

INSTANTIATE_TEST_SUITE_P(AlterColumnMultipleStringSourceTest,
                         AlterColumnMultipleStringSourceTest,
                         ::testing::Values("none", "dict"),
                         [](const auto& info) { return info.param; });

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  // Initialize DBHandler in order to ensure that QueryRunner uses the same SysCatalog
  // instance.
  DBHandlerTestFixture::createDBHandler();

  // TODO: Replace use of QueryRunner with DBHandlerTestFixture
  QR::init(BASE_PATH);

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
